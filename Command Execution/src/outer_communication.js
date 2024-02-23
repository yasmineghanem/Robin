const http = require("http");
const vscode = require("vscode");

function activate(context) {
  // Example command registration
  let disposable = vscode.commands.registerCommand(
    "extension.helloWorld",
    function () {
      vscode.window.showInformationMessage("Hello World from your Extension!");
    }
  );

  context.subscriptions.push(disposable);

  // Start a simple HTTP server
  const server = http.createServer((req, res) => {
    if (req.url === "/execute-command") {
      // Execute your VS Code command here
      vscode.commands.executeCommand("robin.helloWorld").then(
        () => {
          res.writeHead(200, { "Content-Type": "text/plain" });
          res.end("Command executed successfully");
        },
        (err) => {
          res.writeHead(500, { "Content-Type": "text/plain" });
          res.end("Failed to execute command");
        }
      );
    } else {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not Found");
    }
  });

  server.listen(3000, () => {
    console.log("Server listening on port 3000");
  });
}

function deactivate() {}

module.exports = { activate, deactivate };
