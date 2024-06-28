import * as vscode from 'vscode';
import { ADD_WHITESPACE, ASSIGN_VARIABLE, CONDITIONAL,ASSERTION, DECLARE_CONSTANT, DECLARE_FUNCTION, DECLARE_VARIABLE, FOR_LOOP, FUNCTION_CALL, GET_AST, IMPORT_LIBRARY, IMPORT_MODULE, NO_ACTIVE_TEXT_EDITOR, OPERATION, WHILE_LOOP, TYPE_CASTING, USER_INPUT, PRINT, LINE_COMMENT, BLOCK_COMMENT } from '../constants/code';
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

// Import
router.post('/import-library',
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            IMPORT_LIBRARY,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Module Import 
router.post('/import-module',
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


// Operation
router.post("/operation",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            OPERATION,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Assertion
router.post("/assertion",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            ASSERTION,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Casting
router.post("/type-casting",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            TYPE_CASTING,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// User input
router.post("/user-input",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            USER_INPUT,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Print
router.post("/print",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            PRINT,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Line comment
router.post("/line-comment",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            LINE_COMMENT,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Block comment
router.post("/block-comment",
    (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            BLOCK_COMMENT,
            data,
            successHandler,
            errorHandler,
            res
        );
    }
);

// Conditional
router.post("/conditional"
    , (req: Request, res: Response) => {
        const data = req.body;
        executeCommand(
            CONDITIONAL,
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