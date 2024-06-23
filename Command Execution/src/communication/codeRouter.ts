import * as vscode from 'vscode';
import { DECLARE_FUNCTION, DECLARE_VARIABLE, GET_AST, NO_ACTIVE_TEXT_EDITOR } from '../constants/code';
import { errorHandler, executeCommand, showError, successHandler } from './utilities';
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
router.post('/declare-variable',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            DECLARE_VARIABLE,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// declare function
router.post('/declare-function',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            DECLARE_FUNCTION,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// get AST
router.get('/ast',
    (req: Request, res: Response) => {
        // const data = req.body;
        executeCommand(
            GET_AST,
            {},
            successHandler,
            errorHandler,
            res
        );
    }
);
export default router;