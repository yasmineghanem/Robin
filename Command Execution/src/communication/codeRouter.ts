import * as vscode from 'vscode';
import { ADD_WHITESPACE, DECLARE_CONSTANT, DECLARE_FUNCTION, DECLARE_VARIABLE, FOR_LOOP, FUNCTION_CALL, GET_AST, NO_ACTIVE_TEXT_EDITOR, WHILE_LOOP } from '../constants/code';
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

// function call
router.post('/function-call',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            FUNCTION_CALL,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// declare constant
router.post('/declare-constant',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            DECLARE_CONSTANT,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

router.post("/for-loop",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            FOR_LOOP,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

router.post("/while-loop",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            WHILE_LOOP,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);


// add whitespace
router.get('/add-whitespace',
    (req: Request, res: Response) => {
        executeCommand(
            ADD_WHITESPACE,
            req.query,
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