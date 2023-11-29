// Copyright (c) 2017-2023 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

// This server interacts directly with the GPU, and listens on a UNIX socket for clients
// of the Dawn Wire protocol.

#include "workerd/api/gpu/voodoo/voodoo-server.h"
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>
#include <dawn/wire/WireServer.h>
#include <filesystem>
#include <kj/async-io.h>
#include <kj/debug.h>
#include <kj/main.h>
#include <unistd.h>
#include <workerd/api/gpu/voodoo/voodoo-pipe.h>
#include <workerd/api/gpu/voodoo/voodoo-protocol.h>

class VoodooMain : public kj::TaskSet::ErrorHandler {
public:
  VoodooMain(kj::ProcessContext& context) : context(context) {}

  void taskFailed(kj::Exception&& exception) override {
    KJ_LOG(ERROR, "task failed handling connection", exception);
  }

  kj::MainBuilder::Validity setListenPath(kj::StringPtr path) {
    listenPath = path;
    return true;
  }

  kj::MainBuilder::Validity startServer() {
    KJ_DBG(listenPath, "will start listening server");
    workerd::api::gpu::voodoo::VoodooServer server(listenPath);
    server.startServer();
    return true;
  }

  kj::MainFunc getMain() {
    return kj::MainBuilder(context, "Voodoo GPU handler V0.0",
                           "Exposes a Dawn Wire endpoint on a UNIX socket for dawn clients that "
                           "want to interact with a GPU")
        .expectArg("<listen_path>", KJ_BIND_METHOD(*this, setListenPath))
        .callAfterParsing(KJ_BIND_METHOD(*this, startServer))
        .build();
  }

private:
  kj::StringPtr listenPath;
  kj::ProcessContext& context;
};

KJ_MAIN(VoodooMain)
