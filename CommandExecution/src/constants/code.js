"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.READ_FILE_FAILURE = exports.READ_FILE_SUCCESS = exports.BLOCK_COMMENT_FAILURE = exports.BLOCK_COMMENT_SUCCESS = exports.LINE_COMMENT_FAILURE = exports.LINE_COMMENT_SUCCESS = exports.PRINT_FAILURE = exports.PRINT_SUCCESS = exports.USER_INPUT_FAILURE = exports.USER_INPUT_SUCCESS = exports.CASTING_FAILURE = exports.CASTING_SUCCESS = exports.ASSERTION_FAILURE = exports.ASSERTION_SUCCESS = exports.IMPORT_FAILURE = exports.IMPORT_SUCCESS = exports.ASSIGNMENT_FAILURE = exports.ASSIGNMENT_SUCCESS = exports.FUNCTION_CALL_FAILURE = exports.FUNCTION_CALL_SUCCESS = exports.FUNCTION_FAILURE = exports.FUNCTION_SUCCESS = exports.FILE_EXT_FAILURE = exports.VARIABLE_FAILURE = exports.VARIABLE_SUCCESS = exports.EXIT_SCOPE = exports.TRY_EXCEPT = exports.DECLARE_CLASS = exports.WRITE_FILE = exports.READ_FILE = exports.BLOCK_COMMENT = exports.LINE_COMMENT = exports.PRINT = exports.USER_INPUT = exports.TYPE_CASTING = exports.ASSERTION = exports.ARRAY_OPERATION = exports.CONDITIONAL = exports.OPERATION = exports.WHILE_LOOP = exports.FOR_LOOP = exports.IMPORT_MODULE = exports.IMPORT_LIBRARY = exports.ASSIGN_VARIABLE = exports.ADD_WHITESPACE = exports.DECLARE_CONSTANT = exports.FUNCTION_CALL = exports.GET_AST = exports.DECLARE_FUNCTION = exports.DECLARE_VARIABLE = void 0;
exports.EXIT_SCOPE_FAILURE = exports.EXIT_SCOPE_SUCCESS = exports.OPERATION_FAILURE = exports.OPERATION_SUCCESS = exports.LOOP_FAILURE = exports.LOOP_SUCCESS = exports.WHITE_SPACE_FAILURE = exports.WHITE_SPACE_SUCCESS = exports.NO_ACTIVE_TEXT_EDITOR = exports.TRY_EXCEPT_FAILURE = exports.TRY_EXCEPT_SUCCESS = exports.WRITE_FILE_FAILURE = exports.WRITE_FILE_SUCCESS = void 0;
/**
 * Commands
 */
exports.DECLARE_VARIABLE = "robin.declareVariable";
exports.DECLARE_FUNCTION = "robin.declareFunction";
exports.GET_AST = "robin.getAST";
exports.FUNCTION_CALL = "robin.functionCall";
exports.DECLARE_CONSTANT = "robin.declareConstant";
exports.ADD_WHITESPACE = "robin.addWhitespace";
exports.ASSIGN_VARIABLE = "robin.assignVariable";
exports.IMPORT_LIBRARY = "robin.importLibrary";
exports.IMPORT_MODULE = "robin.importModule";
exports.FOR_LOOP = "robin.forLoop";
exports.WHILE_LOOP = "robin.whileLoop";
exports.OPERATION = "robin.operation";
exports.CONDITIONAL = "robin.conditional";
exports.ARRAY_OPERATION = "robin.arrayOperation";
exports.ASSERTION = "robin.assertion";
exports.TYPE_CASTING = "robin.typeCasting";
exports.USER_INPUT = "robin.userInput";
exports.PRINT = "robin.print";
exports.LINE_COMMENT = "robin.lineComment";
exports.BLOCK_COMMENT = "robin.blockComment";
exports.READ_FILE = "robin.readFile";
exports.WRITE_FILE = "robin.writeFile";
exports.DECLARE_CLASS = "robin.declareClass";
exports.TRY_EXCEPT = "robin.tryExcept";
exports.EXIT_SCOPE = 'robin.exitScope';
/**
 * Variable declaration messages
 */
exports.VARIABLE_SUCCESS = "Variable declared successfully";
exports.VARIABLE_FAILURE = "Failed to declare variable";
exports.FILE_EXT_FAILURE = "Unsupported file extension";
/**
 * Function declaration messages
 */
exports.FUNCTION_SUCCESS = "Function declared successfully";
exports.FUNCTION_FAILURE = "Failed to declare function";
exports.FUNCTION_CALL_SUCCESS = "Function call successfully";
exports.FUNCTION_CALL_FAILURE = "Failed to call function";
/**
 * Variable Assignment messages
*/
exports.ASSIGNMENT_SUCCESS = "Variable assigned successfully";
exports.ASSIGNMENT_FAILURE = "Failed to assign variable";
/**
 * Import messages
 */
exports.IMPORT_SUCCESS = "Import successful";
exports.IMPORT_FAILURE = "Failed to import";
/**
 * Assertion
 */
exports.ASSERTION_SUCCESS = "Assertion successful";
exports.ASSERTION_FAILURE = "Failed to assert";
/**
 * Casting
*/
exports.CASTING_SUCCESS = "Casting successful";
exports.CASTING_FAILURE = "Failed to cast";
/**
 *  User Input
*/
exports.USER_INPUT_SUCCESS = "User input successful";
exports.USER_INPUT_FAILURE = "Failed to get user input";
/**
 * Print
*/
exports.PRINT_SUCCESS = "Print successful";
exports.PRINT_FAILURE = "Failed to print";
/**
 * Line Comment
*/
exports.LINE_COMMENT_SUCCESS = "Line comment successful";
exports.LINE_COMMENT_FAILURE = "Failed to comment line";
/**
 * Block Comment
*/
exports.BLOCK_COMMENT_SUCCESS = "Block comment successful";
exports.BLOCK_COMMENT_FAILURE = "Failed to comment block";
/**
 * Read File
*/
exports.READ_FILE_SUCCESS = "Read file successful";
exports.READ_FILE_FAILURE = "Failed to read file";
/**
 * Write File
*/
exports.WRITE_FILE_SUCCESS = "Write file successful";
exports.WRITE_FILE_FAILURE = "Failed to write file";
/**
 * Try Except
*/
exports.TRY_EXCEPT_SUCCESS = "Try Except block successful";
exports.TRY_EXCEPT_FAILURE = "Failed to write try except block";
exports.NO_ACTIVE_TEXT_EDITOR = "No active text editor!";
exports.WHITE_SPACE_SUCCESS = "White space added successfully!";
exports.WHITE_SPACE_FAILURE = "Failed to add white space!";
// loops
exports.LOOP_SUCCESS = "Loop created successfully!";
exports.LOOP_FAILURE = "Failed to create loop!";
// operation
exports.OPERATION_SUCCESS = "Operation created successfully!";
exports.OPERATION_FAILURE = "Failed to create operation!";
// exit scope
exports.EXIT_SCOPE_SUCCESS = "Scope exited successfully!";
exports.EXIT_SCOPE_FAILURE = "Failed to exit scope!";
//# sourceMappingURL=code.js.map