import * as vscode from 'vscode';
import { ADD_WHITESPACE, DECLARE_CONSTANT, DECLARE_FUNCTION, DECLARE_VARIABLE, IMPORT, IMPORT_MODULE , ASSIGN_VARIABLE, FUNCTION_CALL, GET_AST, NO_ACTIVE_TEXT_EDITOR } from '../constants/code';
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

// assign variable
router.post('/assign-variable',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            ASSIGN_VARIABLE,
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

// Import
router.post('/import',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            IMPORT,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Module Import 
router.post('/module-import',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            IMPORT_MODULE,
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