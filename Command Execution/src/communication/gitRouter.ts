import * as vscode from 'vscode';
import {GIT_DISCARD, GIT_PULL, GIT_PUSH, GIT_STAGE, GIT_STASH } from '../constants/GIT';
const router = require('express').Router();

// git commit and push
router.get(
    "/push",
    (req: any, res: any) => {

        vscode.commands.executeCommand(GIT_PUSH, req.query).then(
            (response: any) => {
                if (
                    response.success
                ) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Git push done!" }));
                }
                else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: response.message }));
                }
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// git fetch & pull 
router.get(
    "/pull",
    (req: any, res: any) => {
        vscode.commands.executeCommand(GIT_PULL, req.query).then(
            (response: any) => {
                if (
                    response.success
                ) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Git pull done!" }));
                }
                else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: response.message }));
                }
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// git discard
router.get(
    "/discard",
    (req: any, res: any) => {
        vscode.commands.executeCommand(GIT_DISCARD, req.query).then(
            (response: any) => {
                if (
                    response.success
                ) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Git discard done!" }));
                }
                else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: response.message }));
                }
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// git stage
router.get(
    "/stage",
    (req: any, res: any) => {
        vscode.commands.executeCommand(GIT_STAGE, req.query).then(
            (response: any) => {
                if (
                    response.success
                ) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Git stage done!" }));
                }
                else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: response.message }));
                }
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);


// git stash
router.get(
    "/stash",
    (req: any, res: any) => {
        vscode.commands.executeCommand(GIT_STASH, req.query).then(
            (response: any) => {
                if (
                    response.success
                ) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Git stash done!" }));
                }
                else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: response.message }));
                }
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

export default router;