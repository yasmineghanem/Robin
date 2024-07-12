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
const vscode = __importStar(require("vscode"));
const GIT_1 = require("../constants/GIT");
const router = require('express').Router();
// git commit and push
router.get("/push", (req, res) => {
    vscode.commands.executeCommand(GIT_1.GIT_PUSH, req.query).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git push done!" }));
        }
        else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// git fetch & pull 
router.get("/pull", (req, res) => {
    vscode.commands.executeCommand(GIT_1.GIT_PULL, req.query).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git pull done!" }));
        }
        else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// git discard
router.get("/discard", (req, res) => {
    vscode.commands.executeCommand(GIT_1.GIT_DISCARD, req.query).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git discard done!" }));
        }
        else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// git stage
router.get("/stage", (req, res) => {
    vscode.commands.executeCommand(GIT_1.GIT_STAGE, req.query).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git stage done!" }));
        }
        else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// git stash
router.get("/stash", (req, res) => {
    vscode.commands.executeCommand(GIT_1.GIT_STASH, req.query).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Git stash done!" }));
        }
        else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
exports.default = router;
//# sourceMappingURL=gitRouter.js.map