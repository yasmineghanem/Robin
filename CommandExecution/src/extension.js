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
exports.deactivate = exports.activate = void 0;
// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = __importStar(require("vscode"));
const server_1 = __importDefault(require("./communication/server"));
const commands_1 = __importDefault(require("./commands/commands"));
const path_1 = __importDefault(require("path"));
require('dotenv').config({
    path: path_1.default.resolve(__dirname, '../.env')
}); // Load environment variables from .env
// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
function activate(context) {
    // register all commands
    (0, commands_1.default)();
    server_1.default.listen(process.env.SERVER_HOST_PORT || 3000, () => {
        // activate extension by executing hello world command
        vscode.commands.executeCommand("robin.activate");
        console.log("Server listening on port " + process.env.SERVER_HOST_PORT || 3000);
        // on any request log
        server_1.default.use((req, res, next) => {
            console.log(req.method, req.url);
            next();
        });
    });
}
exports.activate = activate;
// This method is called when your extension is deactivated
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map