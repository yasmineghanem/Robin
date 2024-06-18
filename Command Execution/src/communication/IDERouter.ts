import * as vscode from 'vscode';
const router = require('express').Router();


router.post(
    "/go-to-line",
    (req: any, res: any) => {
        const data = req.body;
        vscode.commands.executeCommand("robin.goToLine", data).then(
            (response: any) => {
                if (response.success) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Navigated to line!" }));
                } else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "No active text editor" }));
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