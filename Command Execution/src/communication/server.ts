import * as vscode from "vscode";
import express from "express";
import cors from "cors";
import fileSystemRouter from "./fileSystemRouter";

const server = express();
// middleware
server.use(cors());
server.use(express.json());


server.use('/file-system',fileSystemRouter);


// endpoints
server.post("/declare-var", (req, res) => {
  const data = req.body;
  const args = req.query;
  vscode.commands.executeCommand("robin.declareVariable", args, data).then(
    () => {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("Command executed successfully");
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("Failed to execute command");
    }
  );
});
server.post("/declare-func", (req, res) => {
  const data = req.body;
  const args = req.query;
  vscode.commands.executeCommand("robin.declareFunction", args, data).then(
    () => {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("Command executed successfully");
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("Failed to execute command");
    }
  );
});
server.post("/declare-class", (req, res) => {
  const data = req.body;
  const args = req.query;
  vscode.commands.executeCommand("robin.declareClass", args, data).then(
    () => {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("Command executed successfully");
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end(err);
    }
  );
});

server.get("/file-name", (req, res) => {
  vscode.commands.executeCommand("robin.fileName").then(
    (name) => {
      // reply with json and 200
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ name }));
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});

server.post("/go-to-line", (req, res) => {
  const data = req.body;
  vscode.commands.executeCommand("robin.goToLocation", data).then(
    () => {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("Command executed successfully");
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("Failed to execute command");
    }
  );
});

// export the server
export default server;
