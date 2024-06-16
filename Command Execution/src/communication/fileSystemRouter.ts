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


// copy file
router.post(
  "/copy-file",
  (req: any, res: any) => {
    const data = req.body;
    vscode.commands.executeCommand("robin.copyFile", data).then(
      (response: any) => {
        if (response.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "File copied successfully!" }));
        }
        else {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "File not found" }));
        }
      },
      (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
      }
    );
  }
);


// copy directory
router.post(
  "/copy-directory",
  (req: any, res: any) => {
    const data = req.body;
    vscode.commands.executeCommand("robin.copyDirectory", data).then(
      (response: any) => {
        if (response.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "Directory copied successfully!" }));
        }
        else {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "Directory not found" }));
        }
      },
      (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
      }
    );
  }
);


// delete file or directory
router.post(
  "/delete",
  (req: any, res: any) => {
    const data = req.body;
    vscode.commands.executeCommand("robin.deleteFile", data).then(
      (response: any) => {
        if (response.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "File/Directory deleted successfully!" }));
        }
        else {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "File/Directory not found" }));
        }
      },
      (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
      }
    );
  }
);

// rename file or directory
router.post(
  "/rename",
  (req: any, res: any) => {
    const data = req.body;
    vscode.commands.executeCommand("robin.rename", data).then(
      (response: any) => {
        if (response.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "File/Directory renamed successfully!" }));
        }
        else {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "File/Directory not found" }));
        }
      },
      (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
      }
    );
  }
);

// copy to clipboard
// router.post(
//   "/copy-clipboard",
//   (req: any, res: any) => {
//     const data = req.body;
//     vscode.commands.executeCommand("robin.copyFileClipboard", data).then(
//       (response: any) => {
//         if (response.success) {
//           res.writeHead(200, { "Content-Type": "application/json" });
//           res.end(JSON.stringify({ message: "File copied to clipboard!" }));
//         }
//         else {
//           res.writeHead(404, { "Content-Type": "application/json" });
//           res.end(JSON.stringify({ message: "File not found" }));
//         }
//       },
//       (err) => {
//         res.writeHead(500, { "Content-Type": "application/json" });
//         res.end(JSON.stringify({ error: err }));
//       }
//     );
//   }
// );

// save
router.post(
  "/save",
  (req: any, res: any) => {
    const data = req.body;
    vscode.commands.executeCommand("robin.save", data).then(
      (response: any) => {
        if (response.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "File saved successfully!" }));
        }
        else {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ message: "No active files found" }));
        }
      },
      (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
      }
    );
  }
);
export default router;