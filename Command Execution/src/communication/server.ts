import * as vscode from "vscode";
import express from "express";
import cors from "cors";
import fileSystemRouter from "./fileSystemRouter";
import IDERouter from "./IDERouter";
import codeRouter from "./codeRouter";
// automatically pick platform
const say = require('say');

const server = express();
// middleware
server.use(cors());
server.use(express.json());


server.use('/file-system', fileSystemRouter);
server.use('/ide', IDERouter);
server.use('/code', codeRouter);


// middleware to modify responses before sending them
// server.use((req, res: any, next) => {
//   // speak the message 
//   say.speak(
//     res?.message,
//     'Microsoft Zira Desktop',
//     // 'Good News',
//     1.5,
//     (err: any) => {
//       if (err) {
//         return console.error(err)
//       }
//       console.log('Text has been spoken.')
//     }

//   )




//   next();
// });

// endpoints
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
