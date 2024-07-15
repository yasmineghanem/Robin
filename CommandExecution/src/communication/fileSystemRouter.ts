import * as vscode from "vscode";
import {
  COPY_DIRECTORY,
  COPY_FILE,
  CREATE_DIRECTORY,
  CREATE_FILE,
  DELETE_FILE,
  RENAME,
  SAVE,
  GET_FILES,
} from "../constants/fileSystem";
const router = require("express").Router();

router.post("/create-file", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand(CREATE_FILE, data).then(
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
  vscode.commands.executeCommand(CREATE_DIRECTORY, data).then(
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
router.post("/copy-file", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand(COPY_FILE, data).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "File copied successfully!" }));
      } else {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "File not found" }));
      }
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});

// copy directory
router.post("/copy-directory", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand(COPY_DIRECTORY, data).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Directory copied successfully!" }));
      } else {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Directory not found" }));
      }
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});

// delete file or directory
router.post("/delete", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand(DELETE_FILE, data).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({ message: "File/Directory deleted successfully!" })
        );
      } else {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "File/Directory not found" }));
      }
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});

// rename file or directory
router.post("/rename", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand(RENAME, data).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({ message: "File/Directory renamed successfully!" })
        );
      } else {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "File/Directory not found" }));
      }
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});

// save
router.post("/save", (req: any, res: any) => {
  const data = req.body;
  vscode.commands.executeCommand(SAVE, data).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "File saved successfully!" }));
      } else {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "No active files found" }));
      }
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});

// get files in workspace
router.post("/get-files", (req: any, res: any) => {
  // const data = req.body;
  vscode.commands.executeCommand(GET_FILES).then(
    (response: any) => {
      if (response.success) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            message: "Retrieved all files successfully in the workspace!",
            files: response.files,
          })
        );
      } else {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "No files found" }));
      }
    },
    (err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err }));
    }
  );
});

// general handling
router.post("/", (req: any, res: any) => {
  const data = req.body;
  const action = data["action"];
  switch (action) {
    case "create":
      if (!data["name"] && !data["fileName"]) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Name is required" }));
      }
      vscode.commands.executeCommand(CREATE_FILE, data).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
          } else {
            res.writeHead(409, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Already exists!" }));
          }
        },
        (err) => {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err }));
        }
      );
      break;
    case "copy-file":
      if (!data["name"] && !data["source"]) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Name is required" }));
      }
      vscode.commands.executeCommand(COPY_FILE, data).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File copied successfully!" }));
          } else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File not found" }));
          }
        },
        (err) => {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err }));
        }
      );
      break;

    case "copy-directory":
      if (!data["name"] && !data["source"]) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Name is required" }));
      }
      vscode.commands.executeCommand(COPY_DIRECTORY, data).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({ message: "Directory copied successfully!" })
            );
          } else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Directory not found" }));
          }
        },
        (err) => {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err }));
        }
      );
      break;

    case "delete":
      vscode.commands.executeCommand(DELETE_FILE, data).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({
                message: "File/Directory deleted successfully!",
              })
            );
          } else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File/Directory not found" }));
          }
        },
        (err) => {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err }));
        }
      );
      break;

    case "rename":
      vscode.commands.executeCommand(RENAME, data).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({
                message: "File/Directory renamed successfully!",
              })
            );
          } else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File/Directory not found" }));
          }
        },
        (err) => {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err }));
        }
      );
      break;

    case "save":
      vscode.commands.executeCommand(SAVE, data).then(
        (response: any) => {
          if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File saved successfully!" }));
          } else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "No active files found" }));
          }
        },
        (err) => {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err }));
        }
      );
      break;

    default:
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ message: "Invalid action" }));
      break;
  }
});

export default router;
