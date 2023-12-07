// Copyright (c) 2017-2022 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

#include "worker-entrypoint.h"
#include "io-context.h"
#include <capnp/message.h>
#include <workerd/jsg/jsg.h>
#include <workerd/api/global-scope.h>
#include <workerd/api/util.h>
#include <workerd/util/sentry.h>
#include <workerd/util/thread-scopes.h>
#include <kj/compat/http.h>

namespace workerd {

namespace {
// Wrapper around a Worker that handles receiving a new event from the outside. In particular,
// this handles:
// - Creating a IoContext and making it current.
// - Executing the worker under lock.
// - Catching exceptions and converting them to HTTP error responses.
//   - Or, falling back to proxying if passThroughOnException() was used.
// - Finish waitUntil() tasks.
class WorkerEntrypoint final: public WorkerInterface {
public:
  // Call this instead of the constructor. It actually adds a wrapper object around the
  // `WorkerEntrypoint`, but the wrapper still implements `WorkerInterface`.
  //
  // WorkerEntrypoint will create a IoContext, and that IoContext may outlive the
  // WorkerEntrypoint by means of a waitUntil() task. Any object(s) which must be kept alive to
  // support the worker for the lifetime of the IoContext (e.g., subsequent pipeline stages)
  // must be passed in via `ioContextDependency`.
  //
  // If this is NOT a zone worker, then `zoneDefaultWorkerLimits` should be a default instance of
  // WorkerLimits::Reader. Hence this is not necessarily the same as
  // topLevelRequest.getZoneDefaultWorkerLimits(), since the top level request may be shared between
  // zone and non-zone workers.
  static kj::Own<WorkerInterface> construct(
                   ThreadContext& threadContext,
                   kj::Own<const Worker> worker,
                   kj::Maybe<kj::StringPtr> entrypointName,
                   kj::Maybe<kj::Own<Worker::Actor>> actor,
                   kj::Own<LimitEnforcer> limitEnforcer,
                   kj::Own<void> ioContextDependency,
                   kj::Own<IoChannelFactory> ioChannelFactory,
                   kj::Own<RequestObserver> metrics,
                   kj::TaskSet& waitUntilTasks,
                   bool tunnelExceptions,
                   kj::Maybe<kj::Own<WorkerTracer>> workerTracer,
                   kj::Maybe<kj::String> cfBlobJson);

  kj::Promise<void> request(
      kj::HttpMethod method, kj::StringPtr url, const kj::HttpHeaders& headers,
      kj::AsyncInputStream& requestBody, Response& response) override;
  kj::Promise<void> connect(kj::StringPtr host, const kj::HttpHeaders& headers,
      kj::AsyncIoStream& connection, ConnectResponse& response,
      kj::HttpConnectSettings settings) override;
  void prewarm(kj::StringPtr url) override;
  kj::Promise<ScheduledResult> runScheduled(kj::Date scheduledTime, kj::StringPtr cron) override;
  kj::Promise<AlarmResult> runAlarm(kj::Date scheduledTime) override;
  kj::Promise<bool> test() override;
  kj::Promise<CustomEvent::Result> customEvent(kj::Own<CustomEvent> event) override;

private:
  class ResponseSentTracker;

  // Members initialized at startup.

  ThreadContext& threadContext;
  kj::TaskSet& waitUntilTasks;
  kj::Maybe<kj::Own<IoContext::IncomingRequest>> incomingRequest;
  bool tunnelExceptions;
  kj::Maybe<kj::StringPtr> entrypointName;
  kj::Maybe<kj::String> cfBlobJson;

  // Hacky members used to hold some temporary state while processing a request.
  // See gory details in WorkerEntrypoint::request().

  kj::Maybe<kj::Promise<void>> proxyTask;
  kj::Maybe<kj::Own<WorkerInterface>> failOpenService;
  bool loggedExceptionEarlier = false;

  void init(
      kj::Own<const Worker> worker,
      kj::Maybe<kj::Own<Worker::Actor>> actor,
      kj::Own<LimitEnforcer> limitEnforcer,
      kj::Own<void> ioContextDependency,
      kj::Own<IoChannelFactory> ioChannelFactory,
      kj::Own<RequestObserver> metrics,
      kj::Maybe<kj::Own<WorkerTracer>> workerTracer);

  template <typename T>
  kj::Promise<T> maybeAddGcPassForTest(IoContext& context, kj::Promise<T> promise);

  kj::Promise<WorkerEntrypoint::AlarmResult> runAlarmImpl(
      kj::Own<IoContext::IncomingRequest> incomingRequest, kj::Date scheduledTime);

public:  // For kj::heap() only; pretend this is private.
  WorkerEntrypoint(kj::Badge<WorkerEntrypoint> badge,
                   ThreadContext& threadContext,
                   kj::TaskSet& waitUntilTasks,
                   bool tunnelExceptions,
                   kj::Maybe<kj::StringPtr> entrypointName,
                   kj::Maybe<kj::String> cfBlobJson);
};

// Simple wrapper around `HttpService::Response` to let us know if the response was sent
// already.
class WorkerEntrypoint::ResponseSentTracker final: public kj::HttpService::Response {
public:
  ResponseSentTracker(kj::HttpService::Response& inner)
      : inner(inner) {}
  KJ_DISALLOW_COPY_AND_MOVE(ResponseSentTracker);

  bool isSent() const { return sent; }

  kj::Own<kj::AsyncOutputStream> send(
      uint statusCode, kj::StringPtr statusText, const kj::HttpHeaders& headers,
      kj::Maybe<uint64_t> expectedBodySize = kj::none) override {
    sent = true;
    return inner.send(statusCode, statusText, headers, expectedBodySize);
  }

  kj::Own<kj::WebSocket> acceptWebSocket(const kj::HttpHeaders& headers) override {
    sent = true;
    return inner.acceptWebSocket(headers);
  }

private:
  kj::HttpService::Response& inner;
  bool sent = false;
};

kj::Own<WorkerInterface> WorkerEntrypoint::construct(
                                   ThreadContext& threadContext,
                                   kj::Own<const Worker> worker,
                                   kj::Maybe<kj::StringPtr> entrypointName,
                                   kj::Maybe<kj::Own<Worker::Actor>> actor,
                                   kj::Own<LimitEnforcer> limitEnforcer,
                                   kj::Own<void> ioContextDependency,
                                   kj::Own<IoChannelFactory> ioChannelFactory,
                                   kj::Own<RequestObserver> metrics,
                                   kj::TaskSet& waitUntilTasks,
                                   bool tunnelExceptions,
                                   kj::Maybe<kj::Own<WorkerTracer>> workerTracer,
                                   kj::Maybe<kj::String> cfBlobJson) {
  auto obj = kj::heap<WorkerEntrypoint>(kj::Badge<WorkerEntrypoint>(), threadContext,
      waitUntilTasks, tunnelExceptions, entrypointName, kj::mv(cfBlobJson));
  obj->init(kj::mv(worker), kj::mv(actor), kj::mv(limitEnforcer),
      kj::mv(ioContextDependency), kj::mv(ioChannelFactory), kj::addRef(*metrics),
      kj::mv(workerTracer));
  auto& wrapper = metrics->wrapWorkerInterface(*obj);
  return kj::attachRef(wrapper, kj::mv(obj), kj::mv(metrics));
}

WorkerEntrypoint::WorkerEntrypoint(kj::Badge<WorkerEntrypoint> badge,
                                   ThreadContext& threadContext,
                                   kj::TaskSet& waitUntilTasks,
                                   bool tunnelExceptions,
                                   kj::Maybe<kj::StringPtr> entrypointName,
                                   kj::Maybe<kj::String> cfBlobJson)
    : threadContext(threadContext),
      waitUntilTasks(waitUntilTasks),
      tunnelExceptions(tunnelExceptions),
      entrypointName(entrypointName),
      cfBlobJson(kj::mv(cfBlobJson)) {}

void WorkerEntrypoint::init(
    kj::Own<const Worker> worker,
    kj::Maybe<kj::Own<Worker::Actor>> actor,
    kj::Own<LimitEnforcer> limitEnforcer,
    kj::Own<void> ioContextDependency,
    kj::Own<IoChannelFactory> ioChannelFactory,
    kj::Own<RequestObserver> metrics,
    kj::Maybe<kj::Own<WorkerTracer>> workerTracer) {
  // We need to construct the IoContext -- unless this is an actor and it already has a
  // IoContext, in which case we reuse it.

  auto newContext = [&]() {
    auto actorRef = actor.map([](kj::Own<Worker::Actor>& ptr) -> Worker::Actor& {
      return *ptr;
    });

    return kj::refcounted<IoContext>(
        threadContext, kj::mv(worker), actorRef, kj::mv(limitEnforcer))
            .attach(kj::mv(ioContextDependency));
  };

  kj::Own<IoContext> context;
  KJ_IF_SOME(a, actor) {
    KJ_IF_SOME(rc, a.get()->getIoContext()) {
      context = kj::addRef(rc);
    } else {
      context = newContext();
      a.get()->setIoContext(kj::addRef(*context));
    }
  } else {
    context = newContext();
  }

  incomingRequest = kj::heap<IoContext::IncomingRequest>(
      kj::mv(context), kj::mv(ioChannelFactory), kj::mv(metrics),
      kj::mv(workerTracer))
      .attach(kj::mv(actor));
}

kj::Promise<void> WorkerEntrypoint::request(
    kj::HttpMethod method, kj::StringPtr url, const kj::HttpHeaders& headers,
    kj::AsyncInputStream& requestBody, Response& response) {
  auto incomingRequest = kj::mv(KJ_REQUIRE_NONNULL(this->incomingRequest,
                                "request() can only be called once"));
  this->incomingRequest = kj::none;
  incomingRequest->delivered();
  auto& context = incomingRequest->getContext();

  auto wrappedResponse = kj::heap<ResponseSentTracker>(response);

  bool isActor = context.getActor() != kj::none;

  KJ_IF_SOME(t, incomingRequest->getWorkerTracer()) {
    auto timestamp = context.now();
    kj::String cfJson;
    KJ_IF_SOME(c, cfBlobJson) {
      cfJson = kj::str(c);
    }

    // To match our historical behavior (when we used to pull the headers from the JavaScript
    // object later on), we need to canonicalize the headers, including:
    // - Lower-case the header name.
    // - Combine multiple headers with the same name into a comma-delimited list. (This explicitly
    //   breaks the Set-Cookie header, incidentally, but should be equivalent for all other
    //   headers.)
    kj::TreeMap<kj::String, kj::Vector<kj::StringPtr>> traceHeaders;
    headers.forEach([&](kj::StringPtr name, kj::StringPtr value) {
      kj::String lower = api::toLower(name);
      auto& slot = traceHeaders.findOrCreate(lower,
          [&]() { return decltype(traceHeaders)::Entry {kj::mv(lower), {}}; });
      slot.add(value);
    });
    auto traceHeadersArray = KJ_MAP(entry, traceHeaders) {
      return Trace::FetchEventInfo::Header(kj::mv(entry.key),
          kj::strArray(entry.value, ", "));
    };

    t.setEventInfo(timestamp, Trace::FetchEventInfo(method, kj::str(url),
        kj::mv(cfJson), kj::mv(traceHeadersArray)));
  }

  auto metricsForCatch = kj::addRef(incomingRequest->getMetrics());

  return context.run(
      [this, &context, method, url, &headers, &requestBody,
       &metrics = incomingRequest->getMetrics(),
       &wrappedResponse = *wrappedResponse, entrypointName = entrypointName]
      (Worker::Lock& lock) mutable {
    jsg::AsyncContextFrame::StorageScope traceScope = context.makeAsyncTraceScope(lock);

    return lock.getGlobalScope().request(
        method, url, headers, requestBody, wrappedResponse,
        cfBlobJson, lock, lock.getExportedHandler(entrypointName, context.getActor()));
  }).then([this](api::DeferredProxy<void> deferredProxy) {
    proxyTask = kj::mv(deferredProxy.proxyTask);
  }).exclusiveJoin(context.onAbort())
      .catch_([this,&context](kj::Exception&& exception) mutable -> kj::Promise<void> {
    // Log JS exceptions to the JS console, if fiddle is attached. This also has the effect of
    // logging internal errors to syslog.
    loggedExceptionEarlier = true;
    context.logUncaughtExceptionAsync(UncaughtExceptionSource::REQUEST_HANDLER,
                                      kj::cp(exception));

    // Do not allow the exception to escape the isolate without waiting for the output gate to
    // open. Note that in the success path, this is taken care of in `FetchEvent::respondWith()`.
    return context.waitForOutputLocks()
        .then([exception = kj::mv(exception)]() mutable
              -> kj::Promise<void> { return kj::mv(exception); });
  }).attach(kj::defer([this,incomingRequest = kj::mv(incomingRequest),&context]() mutable {
    // The request has been canceled, but allow it to continue executing in the background.
    if (context.isFailOpen()) {
      // Fail-open behavior has been chosen, we'd better save an interface that we can use for
      // that purpose later.
      failOpenService = context.getSubrequestChannelNoChecks(IoContext::NEXT_CLIENT_CHANNEL, false,
                                                             kj::mv(cfBlobJson));
    }
    auto promise = incomingRequest->drain().attach(kj::mv(incomingRequest));
    waitUntilTasks.add(maybeAddGcPassForTest(context, kj::mv(promise)));
  })).then([this]() -> kj::Promise<void> {
    // Now that the IoContext is dropped (unless it had waitUntil()s), we can finish proxying
    // without pinning it or the isolate into memory.
    KJ_IF_SOME(p, proxyTask) {
      return kj::mv(p);
    } else {
      return kj::READY_NOW;
    }
  }).attach(kj::defer([this]() mutable {
    // If we're being cancelled, we need to make sure `proxyTask` gets canceled.
    proxyTask = kj::none;
  })).catch_([this,wrappedResponse = kj::mv(wrappedResponse),isActor,
              method, url, &headers, &requestBody, metrics = kj::mv(metricsForCatch)]
             (kj::Exception&& exception) mutable -> kj::Promise<void> {
    // Don't return errors to end user.

    auto isInternalException = !jsg::isTunneledException(exception.getDescription())
        && !jsg::isDoNotLogException(exception.getDescription());
    if (!loggedExceptionEarlier) {
      // This exception seems to have originated during the deferred proxy task, so it was not
      // logged to the IoContext earlier.
      if (exception.getType() != kj::Exception::Type::DISCONNECTED && isInternalException) {
        LOG_EXCEPTION("workerEntrypoint", exception);
      } else {
        KJ_LOG(INFO, exception);  // Run with --verbose to see exception logs.
      }
    }

    auto exceptionToPropagate = [&]() {
      if (isInternalException) {
        // We've already logged it here, the only thing that matters to the client is that we failed
        // due to an internal error. Note that this does not need to be labeled "remote." since jsg
        // will sanitize it as an internal error. Note that we use `setDescription()` to preserve
        // the exception type for `cjfs::makeInternalError(...)` downstream.
        exception.setDescription(kj::str(
            "worker_do_not_log; Request failed due to internal error"));
        return kj::mv(exception);
      } else {
        // We do not care how many remote capnp servers this went through since we are returning
        // it to the worker via jsg.
        // TODO(someday) We also do this stripping when making the tunneled exception for
        // `jsg::isTunneledException(...)`. It would be lovely if we could simply store some type
        // instead of `loggedExceptionEarlier`. It would save use some work.
        auto description = jsg::stripRemoteExceptionPrefix(exception.getDescription());
        if (!description.startsWith("remote.")) {
          // If we already were annotated as remote from some other worker entrypoint, no point
          // adding an additional prefix.
          exception.setDescription(kj::str("remote.", description));
        }
        return kj::mv(exception);
      }

    };

    if (wrappedResponse->isSent()) {
      // We can't fail open if the response was already sent, so set `failOpenService` null so that
      // that branch isn't taken below.
      failOpenService = kj::none;
    }

    if (isActor) {
      // We want to tunnel exceptions from actors back to the caller.
      // TODO(cleanup): We'd really like to tunnel exceptions any time a worker is calling another
      // worker, not just for actors (and W2W below), but getting that right will require cleaning
      // up error handling more generally.
      return exceptionToPropagate();
    } else KJ_IF_SOME(service, failOpenService) {
      // Fall back to origin.

      // We're catching the exception, but metrics should still indicate an exception.
      metrics->reportFailure(exception);

      auto promise = kj::evalNow([&] {
        auto promise = service.get()->request(
            method, url, headers, requestBody, *wrappedResponse);
        metrics->setFailedOpen(true);
        return promise.attach(kj::mv(service));
      });
      return promise.catch_([this,wrappedResponse = kj::mv(wrappedResponse),
                             metrics = kj::mv(metrics)]
                            (kj::Exception&& e) mutable {
        metrics->setFailedOpen(false);
        if (e.getType() != kj::Exception::Type::DISCONNECTED &&
            // Avoid logging recognized external errors here, such as invalid headers returned from
            // the server.
            !jsg::isTunneledException(e.getDescription()) &&
            !jsg::isDoNotLogException(e.getDescription())) {
          LOG_EXCEPTION("failOpenFallback", e);
        }
        if (!wrappedResponse->isSent()) {
          kj::HttpHeaders headers(threadContext.getHeaderTable());
          wrappedResponse->send(500, "Internal Server Error", headers, uint64_t(0));
        }
      });
    } else if (tunnelExceptions) {
      // Like with the isActor check, we want to return exceptions back to the caller.
      // We don't want to handle this case the same as the isActor case though, since we want
      // fail-open to operate normally, which means this case must happen after fail-open handling.
      return exceptionToPropagate();
    } else {
      // Return error.

      // We're catching the exception and replacing it with 5xx, but metrics should still indicate
      // an exception.
      metrics->reportFailure(exception);

      // We can't send an error response if a response was already started; we can only drop the
      // connection in that case.
      if (!wrappedResponse->isSent()) {
        kj::HttpHeaders headers(threadContext.getHeaderTable());
        if (exception.getType() == kj::Exception::Type::OVERLOADED) {
          wrappedResponse->send(503, "Service Unavailable", headers, uint64_t(0));
        } else {
          wrappedResponse->send(500, "Internal Server Error", headers, uint64_t(0));
        }
      }

      return kj::READY_NOW;
    }
  });
}

kj::Promise<void> WorkerEntrypoint::connect(kj::StringPtr host, const kj::HttpHeaders& headers,
    kj::AsyncIoStream& connection, ConnectResponse& response,
    kj::HttpConnectSettings settings) {
  JSG_FAIL_REQUIRE(TypeError, "Incoming CONNECT on a worker not supported");
}

void WorkerEntrypoint::prewarm(kj::StringPtr url) {
  // Nothing to do, the worker is already loaded.

  auto incomingRequest = kj::mv(KJ_REQUIRE_NONNULL(this->incomingRequest,
                                "prewarm() can only be called once"));
  incomingRequest->getMetrics().setIsPrewarm();

  // Intentionally don't call incomingRequest->delivered() for prewarm requests.

  // TODO(someday): Ideally, middleware workers would forward prewarm() to the next stage. At
  //   present we don't have a good way to decide what stage that is, especially given that we'll
  //   be switching to `next` being a binding in the future.
}

kj::Promise<WorkerInterface::ScheduledResult> WorkerEntrypoint::runScheduled(
    kj::Date scheduledTime,
    kj::StringPtr cron) {
  auto incomingRequest = kj::mv(KJ_REQUIRE_NONNULL(this->incomingRequest,
                                "runScheduled() can only be called once"));
  this->incomingRequest = kj::none;
  incomingRequest->delivered();
  auto& context = incomingRequest->getContext();

  KJ_ASSERT(context.getActor() == kj::none);
  // This code currently doesn't work with actors because cancellations occur immediately, without
  // calling context->drain(). We don't ever send scheduled events to actors. If we do, we'll have
  // to think more about this.

  KJ_IF_SOME(t, context.getWorkerTracer()) {
    double eventTime = (scheduledTime - kj::UNIX_EPOCH) / kj::MILLISECONDS;
    t.setEventInfo(context.now(), Trace::ScheduledEventInfo(eventTime, kj::str(cron)));
  }

  // Scheduled handlers run entirely in waitUntil() tasks.
  context.addWaitUntil(context.run(
      [scheduledTime, cron, entrypointName=entrypointName, &context,
       &metrics = incomingRequest->getMetrics()]
      (Worker::Lock& lock) mutable {
    jsg::AsyncContextFrame::StorageScope traceScope = context.makeAsyncTraceScope(lock);

    lock.getGlobalScope().startScheduled(scheduledTime, cron, lock,
        lock.getExportedHandler(entrypointName, context.getActor()));
  }));

  static auto constexpr waitForFinished = [](IoContext& context,
                                             kj::Own<IoContext::IncomingRequest> request)
      -> kj::Promise<WorkerInterface::ScheduledResult> {
    bool completed = co_await request->finishScheduled();
    co_return WorkerInterface::ScheduledResult {
      .retry = context.shouldRetryScheduled(),
      .outcome = completed ? context.waitUntilStatus() : EventOutcome::EXCEEDED_CPU
    };
  };

  return maybeAddGcPassForTest(context, waitForFinished(context, kj::mv(incomingRequest)));
}

kj::Promise<WorkerInterface::AlarmResult> WorkerEntrypoint::runAlarmImpl(
    kj::Own<IoContext::IncomingRequest> incomingRequest, kj::Date scheduledTime) {
  // We want to de-duplicate alarm requests as follows:
  // - An alarm must not be canceled once it is running, UNLESS the whole actor is shut down.
  // - If multiple alarm invocations arrive with the same scheduled time, we only run one.
  // - If we are asked to schedule an alarm while one is running, we wait for the running alarm to
  //   finish.
  // - However, we schedule no more than one alarm. If another one (with yet another different
  //   scheduled time) arrives while we still have one running and one scheduled, we discard the
  //   previous scheduled alarm.

  auto& context = incomingRequest->getContext();
  auto& actor = KJ_REQUIRE_NONNULL(context.getActor(), "alarm() should only work with actors");

  KJ_IF_SOME(promise, actor.getAlarm(scheduledTime)) {
    // There is a pre-existing alarm for `scheduledTime`, we can just wait for its result.
    // TODO(someday) If the request responsible for fulfilling this alarm were to be cancelled, then
    // we could probably take over and try to fulfill it ourselves. Maybe we'd want to loop on
    // `actor.getAlarm()`? We'd have to distinguish between rescheduling and request cancellation.
    auto result = co_await promise;
    co_return result;
  }

  // There isn't a pre-existing alarm, we can call `delivered()` (and emit metrics events).
  incomingRequest->delivered();

  KJ_IF_SOME(t, incomingRequest->getWorkerTracer()) {
    t.setEventInfo(context.now(), Trace::AlarmEventInfo(scheduledTime));
  }

  auto scheduleAlarmResult = co_await actor.scheduleAlarm(scheduledTime);
  KJ_SWITCH_ONEOF(scheduleAlarmResult) {
    KJ_CASE_ONEOF(af, WorkerInterface::AlarmFulfiller) {
      // We're now in charge of running this alarm!
      auto cancellationGuard = kj::defer([&af]() {
        // Our promise chain was cancelled, let's cancel our fulfiller for any other requests
        // that were waiting on us.
        af.cancel();
      });

      KJ_DEFER({
        // The alarm has finished but allow the request to continue executing in the background.
        waitUntilTasks.add(incomingRequest->drain().attach(kj::mv(incomingRequest)));
      });


      try {
        auto result = co_await context.run(
            [scheduledTime, entrypointName=entrypointName, &context](Worker::Lock& lock){
          jsg::AsyncContextFrame::StorageScope traceScope = context.makeAsyncTraceScope(lock);

          auto handler = lock.getExportedHandler(entrypointName, context.getActor());
          return lock.getGlobalScope().runAlarm(scheduledTime, lock, handler);
        });

        // We succeeded, inform any other entrypoints that may be waiting upon us.
        af.fulfill(result);
        cancellationGuard.cancel();
        co_return result;
      } catch (const kj::Exception& e) {
        // We failed, inform any other entrypoints that may be waiting upon us.
        af.reject(e);
        cancellationGuard.cancel();
        throw;
      }
    }
    KJ_CASE_ONEOF(result, WorkerInterface::AlarmResult) {
      // The alarm was cancelled while we were waiting to run, go ahead and return the result.
      co_return result;
    }
  }

  KJ_UNREACHABLE;
}

kj::Promise<WorkerInterface::AlarmResult> WorkerEntrypoint::runAlarm(
    kj::Date scheduledTime) {
  auto incomingRequest = kj::mv(KJ_REQUIRE_NONNULL(this->incomingRequest,
                                "runAlarm() can only be called once"));
  this->incomingRequest = kj::none;

  auto& context = incomingRequest->getContext();
  auto promise = runAlarmImpl(kj::mv(incomingRequest), scheduledTime);
  return maybeAddGcPassForTest(context, kj::mv(promise));
}

kj::Promise<bool> WorkerEntrypoint::test() {
  auto incomingRequest = kj::mv(KJ_REQUIRE_NONNULL(this->incomingRequest,
                                "test() can only be called once"));
  this->incomingRequest = kj::none;
  incomingRequest->delivered();

  auto& context = incomingRequest->getContext();

  context.addWaitUntil(context.run(
      [entrypointName=entrypointName, &context, &metrics = incomingRequest->getMetrics()]
      (Worker::Lock& lock) mutable -> kj::Promise<void> {
    jsg::AsyncContextFrame::StorageScope traceScope = context.makeAsyncTraceScope(lock);

    return context.awaitJs(lock, lock.getGlobalScope()
        .test(lock, lock.getExportedHandler(entrypointName, context.getActor())));
  }));

  static auto constexpr waitForFinished = [](IoContext& context,
                                             kj::Own<IoContext::IncomingRequest> request)
      -> kj::Promise<bool> {
    bool completed = co_await request->finishScheduled();
    auto outcome = completed ? context.waitUntilStatus() : EventOutcome::EXCEEDED_CPU;
    co_return outcome == EventOutcome::OK;
  };

  return maybeAddGcPassForTest(context, waitForFinished(context, kj::mv(incomingRequest)));
}

kj::Promise<WorkerInterface::CustomEvent::Result>
    WorkerEntrypoint::customEvent(kj::Own<CustomEvent> event) {
  auto incomingRequest = kj::mv(KJ_REQUIRE_NONNULL(this->incomingRequest,
                                "customEvent() can only be called once"));
  this->incomingRequest = kj::none;

  auto& context = incomingRequest->getContext();
  auto promise = event->run(kj::mv(incomingRequest), entrypointName).attach(kj::mv(event));

  // TODO(cleanup): In theory `context` may have been destroyed by now if `event->run()` dropped
  //   the `incomingRequest` synchronously. No current implementation does that, and
  //   maybeAddGcPassForTest() is a no-op outside of tests, so I'm ignoring the theoretical problem
  //   for now. Otherwise we will need to `atomicAddRef()` the `Worker` at some point earlier on
  //   but I'd like to avoid that in the non-test case.
  return maybeAddGcPassForTest(context, kj::mv(promise));
}

#ifdef KJ_DEBUG
void requestGc(const Worker& worker) {
  jsg::V8StackScope stackScope;
  auto lock = worker.getIsolate().getApi().lock(stackScope);
  lock->requestGcForTesting();
}

template <typename T>
kj::Promise<T> addGcPassForTest(IoContext& context, kj::Promise<T> promise) {
  auto worker = kj::atomicAddRef(context.getWorker());
  if constexpr (kj::isSameType<T, void>()) {
    co_await promise;
    requestGc(*worker);
  } else {
    auto ret = co_await promise;
    requestGc(*worker);
    co_return kj::mv(ret);
  }
}
#endif

template <typename T>
kj::Promise<T> WorkerEntrypoint::maybeAddGcPassForTest(
    IoContext& context, kj::Promise<T> promise) {
#ifdef KJ_DEBUG
  if (isPredictableModeForTest()) {
    return addGcPassForTest(context, kj::mv(promise));
  }
#endif
  return kj::mv(promise);
}

}  // namespace

kj::Own<WorkerInterface> newWorkerEntrypoint(
    ThreadContext& threadContext,
    kj::Own<const Worker> worker,
    kj::Maybe<kj::StringPtr> entrypointName,
    kj::Maybe<kj::Own<Worker::Actor>> actor,
    kj::Own<LimitEnforcer> limitEnforcer,
    kj::Own<void> ioContextDependency,
    kj::Own<IoChannelFactory> ioChannelFactory,
    kj::Own<RequestObserver> metrics,
    kj::TaskSet& waitUntilTasks,
    bool tunnelExceptions,
    kj::Maybe<kj::Own<WorkerTracer>> workerTracer,
    kj::Maybe<kj::String> cfBlobJson) {
  return WorkerEntrypoint::construct(
      threadContext,
      kj::mv(worker),
      kj::mv(entrypointName),
      kj::mv(actor),
      kj::mv(limitEnforcer),
      kj::mv(ioContextDependency),
      kj::mv(ioChannelFactory),
      kj::mv(metrics),
      waitUntilTasks,
      tunnelExceptions,
      kj::mv(workerTracer),
      kj::mv(cfBlobJson));
}

} // namespace workerd
