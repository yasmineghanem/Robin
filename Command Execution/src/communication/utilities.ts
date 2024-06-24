import * as vscode from 'vscode';

import { Response } from "express";

export const successHandler = (response: any, res: Response) => {
    if (response.success) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        if (response) {
            res.end(JSON.stringify(response));
        } else {
            res.end(JSON.stringify({ message: "Operation done successfully" }));
        }
    } else {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ message: response.message }));
    }
};

export const errorHandler = (err: any, res: any) => {
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(err));
};

export const executeCommand = (command: string, data: any, successHandler: any, errorHandler: any, res: Response) => {
    vscode.commands.executeCommand(command, data).then(
        (response: any) => successHandler(response, res),
        (err) => errorHandler(err, res)
    );
};

export const showMessage = (message: string) => {
    vscode.window.showInformationMessage(message);

};

export const showError = (message: string) => {
    vscode.window.showErrorMessage(message);
}