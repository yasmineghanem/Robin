/**
 * Commands
 */
export const DECLARE_VARIABLE = "robin.declareVariable";
export const DECLARE_FUNCTION = "robin.declareFunction";
export const GET_AST = "robin.getAST";
export const FUNCTION_CALL = "robin.functionCall";
export const DECLARE_CONSTANT = "robin.declareConstant"
export const ADD_WHITESPACE = "robin.addWhitespace";
export const ASSIGN_VARIABLE = "robin.assignVariable";
export const IMPORT_LIBRARY = "robin.importLibrary";
export const IMPORT_MODULE = "robin.importModule";
export const FOR_LOOP = "robin.forLoop";
export const WHILE_LOOP = "robin.whileLoop";
export const OPERATION = "robin.operation";
export const CONDITIONAL = "robin.conditional";
export const ARRAY_OPERATION = "robin.arrayOperation";
export const ASSERTION = "robin.assertion";
export const TYPE_CASTING = "robin.typeCasting";
export const USER_INPUT = "robin.userInput";
export const PRINT = "robin.print";
export const LINE_COMMENT = "robin.lineComment";
export const BLOCK_COMMENT = "robin.blockComment";
export const READ_FILE = "robin.readFile";
export const WRITE_FILE = "robin.writeFile";
export const DECLARE_CLASS = "robin.declareClass";
export const TRY_EXCEPT = "robin.tryExcept";

/**
 * Variable declaration messages
 */
export const VARIABLE_SUCCESS = "Variable declared successfully";
export const VARIABLE_FAILURE = "Failed to declare variable";
export const FILE_EXT_FAILURE = "Unsupported file extension";

/**
 * Function declaration messages
 */

export const FUNCTION_SUCCESS = "Function declared successfully";
export const FUNCTION_FAILURE = "Failed to declare function";

export const FUNCTION_CALL_SUCCESS = "Function call successfully";
export const FUNCTION_CALL_FAILURE = "Failed to call function";

/**
 * Variable Assignment messages
*/
export const ASSIGNMENT_SUCCESS = "Variable assigned successfully";
export const ASSIGNMENT_FAILURE = "Failed to assign variable";

/**
 * Import messages
 */
export const IMPORT_SUCCESS = "Import successful";
export const IMPORT_FAILURE = "Failed to import";

/**
 * Assertion
 */
export const ASSERTION_SUCCESS = "Assertion successful";
export const ASSERTION_FAILURE = "Failed to assert";

/**
 * Casting
*/
export const CASTING_SUCCESS = "Casting successful";
export const CASTING_FAILURE = "Failed to cast";

/**
 *  User Input
*/
export const USER_INPUT_SUCCESS = "User input successful";
export const USER_INPUT_FAILURE = "Failed to get user input";

/** 
 * Print
*/
export const PRINT_SUCCESS = "Print successful";
export const PRINT_FAILURE = "Failed to print";

/**
 * Line Comment
*/
export const LINE_COMMENT_SUCCESS = "Line comment successful";
export const LINE_COMMENT_FAILURE = "Failed to comment line";

/** 
 * Block Comment
*/ 
export const BLOCK_COMMENT_SUCCESS = "Block comment successful";
export const BLOCK_COMMENT_FAILURE = "Failed to comment block";

/**
 * Read File
*/
export const READ_FILE_SUCCESS = "Read file successful";
export const READ_FILE_FAILURE = "Failed to read file";

/**
 * Write File
*/
export const WRITE_FILE_SUCCESS = "Write file successful";
export const WRITE_FILE_FAILURE = "Failed to write file";

/**
 * Try Except
*/
export const TRY_EXCEPT_SUCCESS = "Try Except block successful";
export const TRY_EXCEPT_FAILURE = "Failed to write try except block";

export const NO_ACTIVE_TEXT_EDITOR = "No active text editor!";


export const WHITE_SPACE_SUCCESS = "White space added successfully!";
export const WHITE_SPACE_FAILURE = "Failed to add white space!";

// loops
export const LOOP_SUCCESS = "Loop created successfully!";
export const LOOP_FAILURE = "Failed to create loop!";

// operation
export const OPERATION_SUCCESS = "Operation created successfully!";
export const OPERATION_FAILURE = "Failed to create operation!";