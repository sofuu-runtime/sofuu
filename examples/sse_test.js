const { SSEParser } = sofuu;
const sse = new SSEParser();

sse.onMessage = (evt, data) => {
    console.log("Got Event:", evt);
    console.log("Data:", data);
};

// Simulate feeding stream chunks
sse.feed("data: Hello World");
sse.feed("\n\n");
sse.feed("data: ChatGPT\n\n");
