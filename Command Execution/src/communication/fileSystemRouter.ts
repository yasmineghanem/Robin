import * as vscode from 'vscode';
// import server from "./server";
const router = require('express').Router();

// router prefix
// router.prefix = '/file-system';

router.post("/create-file", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand("robin.createFile", data).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "File created successfully!" }));
      } else {
        res.writeHead(409, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "File already exists!" }));
      }
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});
router.post("/create-directory", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand("robin.createDirectory", data).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            // response.path,
            message: "Directory created successfully!",
          })
        );
      } else {
        res.writeHead(409, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Directory already exists!" }));
      }
    },
    (err) => {
      res.writeHead(400, { "Content-Type": "application/json" });
      console.log(err);
      res.end(JSON.stringify(err));
    }
  );
});


export default router;