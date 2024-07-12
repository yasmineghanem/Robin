"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vscode = __importStar(require("vscode"));
const code_1 = require("../constants/code");
const utilities_1 = require("./utilities");
const express_1 = __importDefault(require("express"));
const router = express_1.default.Router();
// middleware to check if there's an active text editor
router.use((req, res, next) => {
    if (!vscode.window.activeTextEditor) {
        (0, utilities_1.errorHandler)({ message: code_1.NO_ACTIVE_TEXT_EDITOR }, res);
        (0, utilities_1.showError)(code_1.NO_ACTIVE_TEXT_EDITOR);
    }
    else {
        next();
    }
});
// declare variable
router.post("/declare-variable", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.DECLARE_VARIABLE, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// assign variable
router.post("/assign-variable", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.ASSIGN_VARIABLE, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// declare function
router.post("/declare-function", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.DECLARE_FUNCTION, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// function call
router.post("/function-call", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.FUNCTION_CALL, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// declare constant
router.post("/declare-constant", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.DECLARE_CONSTANT, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
router.post("/for-loop", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.FOR_LOOP, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
router.post("/while-loop", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.WHILE_LOOP, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// add whitespace
router.get("/add-whitespace", (req, res) => {
    (0, utilities_1.executeCommand)(code_1.ADD_WHITESPACE, req.query, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Import
router.post("/import-library", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.IMPORT_LIBRARY, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Module Import
router.post("/import-module", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.IMPORT_MODULE, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Operation
router.post("/operation", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.OPERATION, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Assertion
router.post("/assertion", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.ASSERTION, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Casting
router.post("/type-casting", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.TYPE_CASTING, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// User input
router.post("/user-input", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.USER_INPUT, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Print
router.post("/print", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.PRINT, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Line comment
router.post("/line-comment", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.LINE_COMMENT, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Block comment
router.post("/block-comment", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.BLOCK_COMMENT, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Read file
router.post("/read-file", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.READ_FILE, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Write file
router.post("/write-file", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.WRITE_FILE, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Conditional
router.post("/conditional", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.CONDITIONAL, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// class
router.post("/declare-class", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.DECLARE_CLASS, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// get AST
router.get("/ast", (req, res) => {
    // const data = req.body;
    (0, utilities_1.executeCommand)(code_1.GET_AST, {}, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Try Except
router.post("/try-except", (req, res) => {
    const data = req.body;
    (0, utilities_1.executeCommand)(code_1.TRY_EXCEPT, data, utilities_1.successHandler, utilities_1.errorHandler, res);
});
// Try Except
router.get("/exit-scope", (req, res) => {
    (0, utilities_1.executeCommand)(code_1.EXIT_SCOPE, {}, utilities_1.successHandler, utilities_1.errorHandler, res);
});
exports.default = router;
//# sourceMappingURL=codeRouter.js.map