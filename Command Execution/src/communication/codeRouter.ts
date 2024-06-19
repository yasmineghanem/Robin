import * as vscode from 'vscode';
import { DECLARE_FUNCTION, DECLARE_VARIABLE } from '../constants/code';
import { errorHandler, executeCommand, successHandler } from './utilities';
import express, { Request, Response } from "express";

const router = express.Router();

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

export default router;