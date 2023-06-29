interface Blob {

}

interface INativeRequest {
  method: string;
  url: string;

  readAllText(): Promise<String>;
}

class RequestImpl {
  public readonly method: string;
  public readonly url: string;

  constructor(readonly support: INativeRequest) {
    this.method = support.method;
    this.url = support.url;
  }

  public async blob(): Promise<Blob> {
    const support = this.support;

    return {
      async text(): Promise<String> {
        return support.readAllText();
      }
    };
  }
}

export function createRequest(support: INativeRequest) {
  return new RequestImpl(support);
}

export class Response {

}
