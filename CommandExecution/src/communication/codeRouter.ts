import * as vscode from "vscode";
import {
  ADD_WHITESPACE,
  ASSIGN_VARIABLE,
  CONDITIONAL,
  ASSERTION,
  DECLARE_CONSTANT,
  DECLARE_FUNCTION,
  DECLARE_VARIABLE,
  FOR_LOOP,
  FUNCTION_CALL,
  GET_AST,
  IMPORT_LIBRARY,
  IMPORT_MODULE,
  NO_ACTIVE_TEXT_EDITOR,
  OPERATION,
  WHILE_LOOP,
  TYPE_CASTING,
  USER_INPUT,
  PRINT,
  LINE_COMMENT,
  BLOCK_COMMENT,
  READ_FILE,
  WRITE_FILE,
  TRY_EXCEPT,
  DECLARE_CLASS,
  EXIT_SCOPE,
} from "../constants/code";
import {
  errorHandler,
  executeCommand,
  showError,
  successHandler,
} from "./utilities";
import express, { Request, Response } from "express";

const router = express.Router();

// middleware to check if there's an active text editor
router.use((req: Request, res: Response, next) => {
  if (!vscode.window.activeTextEditor) {
    errorHandler({ message: NO_ACTIVE_TEXT_EDITOR }, res);
    showError(NO_ACTIVE_TEXT_EDITOR);
  } else {
    next();
  }
});

// declare variable
router.post("/declare-variable", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(DECLARE_VARIABLE, data, successHandler, errorHandler, res);
});

// assign variable
router.post("/assign-variable", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(ASSIGN_VARIABLE, data, successHandler, errorHandler, res);
});

// declare function
router.post("/declare-function", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(DECLARE_FUNCTION, data, successHandler, errorHandler, res);
});

// function call
router.post("/function-call", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(FUNCTION_CALL, data, successHandler, errorHandler, res);
});

// declare constant
router.post("/declare-constant", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(DECLARE_CONSTANT, data, successHandler, errorHandler, res);
});

router.post("/for-loop", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(FOR_LOOP, data, successHandler, errorHandler, res);
});

router.post("/while-loop", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(WHILE_LOOP, data, successHandler, errorHandler, res);
});

// add whitespace
router.get("/add-whitespace", (req: Request, res: Response) => {
  executeCommand(ADD_WHITESPACE, req.query, successHandler, errorHandler, res);
});

// Import
router.post("/import-library", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(IMPORT_LIBRARY, data, successHandler, errorHandler, res);
});

// Module Import
router.post("/import-module", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(IMPORT_MODULE, data, successHandler, errorHandler, res);
});

// Operation
router.post("/operation", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(OPERATION, data, successHandler, errorHandler, res);
});

// Assertion
router.post("/assertion", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(ASSERTION, data, successHandler, errorHandler, res);
});

// Casting
router.post("/type-casting", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(TYPE_CASTING, data, successHandler, errorHandler, res);
});

// User input
router.post("/user-input", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(USER_INPUT, data, successHandler, errorHandler, res);
});

// Print
router.post("/print", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(PRINT, data, successHandler, errorHandler, res);
});

// Line comment
router.post("/line-comment", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(LINE_COMMENT, data, successHandler, errorHandler, res);
});

// Block comment
router.post("/block-comment", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(BLOCK_COMMENT, data, successHandler, errorHandler, res);
});

// Read file
router.post("/read-file", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(READ_FILE, data, successHandler, errorHandler, res);
});

// Write file
router.post("/write-file", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(WRITE_FILE, data, successHandler, errorHandler, res);
});

// Conditional
router.post("/conditional", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(CONDITIONAL, data, successHandler, errorHandler, res);
});

// class
router.post("/declare-class", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(DECLARE_CLASS, data, successHandler, errorHandler, res);
});

// get AST
router.get("/ast", (req: Request, res: Response) => {
  // const data = req.body;
  executeCommand(GET_AST, {}, successHandler, errorHandler, res);
});

// Try Except
router.post("/try-except", (req: Request, res: Response) => {
  const data = req.body;
  executeCommand(TRY_EXCEPT, data, successHandler, errorHandler, res);
});

// Try Except
router.get("/exit-scope", (req: Request, res: Response) => {
  executeCommand(EXIT_SCOPE, {}, successHandler, errorHandler, res);
});

// get errors
router.get("/errors", (req: Request, res: Response) => {
  const editor = vscode.window.activeTextEditor
  if(editor)
    {
      const uri = editor.document.uri;
      const errors = vscode.languages.getDiagnostics(uri);
      /**
       *  [{"severity":"Error","message":"Parsing failed: 'invalid syntax (<unknown>, line 35)'","range":[{"line":34,"character":17},{"line":34,"character":17}],"source":"Pylint","code":{"value":"E0001:syntax-error","target":{"$mid":1,"path":"/en/latest/user_guide/messages/error/syntax-error.html","scheme":"https","authority":"pylint.readthedocs.io"}}},{"severity":"Error","message":"SyntaxError: invalid syntax (file:///c%3A/Users/77/OneDrive/Desktop/New%20folder/robin.py, line 35)","range":[{"line":34,"character":16},{"line":34,"character":17}],"source":"compile"},{"severity":"Information","message":"\"alsarsora\": Unknown word.","range":[{"line":34,"character":7},{"line":34,"character":16}],"source":"cSpell"},{"severity":"Information","message":"\"sarsora\": Unknown word.","range":[{"line":35,"character":0},{"line":35,"character":7}],"source":"cSpell"}]
       */
      // return error type and line number
      const errorList = errors.map((error) => {
        return {
          severity: error.severity,
          message: error.message,
          range: error.range.start.line,
          source: error.source,
        };
      });
      res.send(errorList);
    }
});

export default router;
