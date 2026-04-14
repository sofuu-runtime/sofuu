const { serve } = sofuu;

console.log("Starting server on port 3000...");
serve(3000, (req, res) => {
    console.log("Received request:", req.method, req.url);
    if (req.body) {
        console.log("Body:", req.body);
    }
    res.send("Hello from Sofuu Native HTTP Server!\n");
});
