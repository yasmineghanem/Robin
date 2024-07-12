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
Object.defineProperty(exports, "__esModule", { value: true });
exports.showError = exports.showMessage = exports.executeCommand = exports.errorHandler = exports.successHandler = void 0;
const vscode = __importStar(require("vscode"));
const successHandler = (response, res) => {
    if (response.success) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        if (response) {
            res.end(JSON.stringify(response));
        }
        else {
            res.end(JSON.stringify({ message: "Operation done successfully" }));
        }
    }
    else {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ message: response.message }));
    }
};
exports.successHandler = successHandler;
const errorHandler = (err, res) => {
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(err));
};
exports.errorHandler = errorHandler;
const executeCommand = (command, data, successHandler, errorHandler, res) => {
    vscode.commands.executeCommand(command, data).then((response) => successHandler(response, res), (err) => errorHandler(err, res));
};
exports.executeCommand = executeCommand;
const showMessage = (message) => {
    vscode.window.showInformationMessage(message);
};
exports.showMessage = showMessage;
const showError = (message) => {
    vscode.window.showErrorMessage(message);
};
exports.showError = showError;
//# sourceMappingURL=utilities.js.map