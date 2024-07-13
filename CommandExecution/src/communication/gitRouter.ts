import * as vscode from "vscode";
import {
  GIT_DISCARD,
  GIT_PULL,
  GIT_PUSH,
  GIT_STAGE,
  GIT_STASH,
} from "../constants/GIT";

const router = require("express").Router();

router.post("/", (req: any, res: any) => {
  // depending on action
  // call the right function

  const body = req.body;
  const action = body.action;
  const message = body.message;

  switch (action) {
    case "push":
      vscode.commands.executeCommand(GIT_PUSH, message).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git push done!" }));
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
          }
        },
        (err) => {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify(err));
        }
      );
      break;
    case "commit":
      vscode.commands.executeCommand(GIT_PUSH, message).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git push done!" }));
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
          }
        },
        (err) => {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify(err));
        }
      );
      break;
    case "pull":
      vscode.commands.executeCommand(GIT_PULL, message).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git pull done!" }));
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
          }
        },
        (err) => {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify(err));
        }
      );
      break;
    case "discard":
      vscode.commands.executeCommand(GIT_DISCARD, message).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git discard done!" }));
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
          }
        },
        (err) => {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify(err));
        }
      );
      break;
    case "stage":
      vscode.commands
        .executeCommand(GIT_STAGE, message)
        .then((response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git stage done!" }));
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
          }
        });
      break;
    case "stash":
      vscode.commands.executeCommand(GIT_STASH, message).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git stash done!" }));
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
          }
        },
        (err) => {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify(err));
        }
      );
      break;
    default:
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ message: "Invalid action" }));
      break;
  }
});

export default router;
