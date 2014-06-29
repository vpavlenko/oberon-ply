#!/usr/bin/env python2

reserved_list = (
    'integer', 'boolean', 
    'array', 'of',
    'true', 'false',
    'div', 'mod',
    'and', 'or', 'not',
    'module', 'var',
    'begin', 'end',
    'if', 'then', 'else',
    'while', 'do',
    'read', 'write', 'writeln',
)

tokens = [
    'ADD', 'SUB', 'MUL',
    'EQ', 'NE', 'GT', 'GE', 'LT', 'LE',
    'ASSIGN',
    'LPAREN', 'RPAREN',
    'LBRA', 'RBRA',
    'COMMA', 'COLON', 'SEP', 'FULLSTOP',
    'NAME', 'NUMBER',
]

SCALAR_TYPES = ('BOOLEAN', 'INTEGER')


# Add case insensitiveness to reserved words

from string import ascii_lowercase

for item in reserved_list:
    if isinstance(item, str):
        re = ''
        for symbol in item:
            if symbol in ascii_lowercase:
                re += '[' + symbol + symbol.upper() + ']'
            else:
                re += symbol
        tokens.append(item.upper())
        globals()['t_' + item.upper()] = re
        

# Tokens

t_ADD      = r'\+'
t_SUB      = r'-'
t_MUL      = r'\*'

t_EQ       = r'='
t_NE       = r'<>'
t_GT       = r'<'
t_GE       = r'<='
t_LT       = r'>'
t_LE       = r'>='

t_ASSIGN   = r':='

t_LPAREN   = r'\('
t_RPAREN   = r'\)'

t_LBRA     = r'\['
t_RBRA     = r'\]'

t_COMMA    = r','
t_SEP      = r';'
t_COLON    = r':'
t_FULLSTOP = r'\.'


class name_token_value:
    def __init__(self, value, type, lineno):
        self.value = value
        self.type = type    # deprecated
        self.lineno = lineno

    def eval(self, environment):
        try:
            result = environment.variables[self.value]
            if result.value is None:
                runtime_error(self.lineno, "Attempt to use value of uninitialized variable '%s'" % (self.value))
            return result
        except LookupError:
            runtime_error(self.lineno, "Undefined variable '%s' in scope '%s'" % (self.value, environment.scope))

    def set(self, environment, new_value):
        if self.value in environment.variables:
            if environment.variables[self.value].type == new_value.type:
                environment.variables[self.value].value = new_value.value
            else:
                runtime_error(self.lineno, 
                    "Type mismatch in assignment to variable '%s': value of type %s expected, value of %s found" %
                    (self.value, environment.variables[self.value].type, new_value.type))
        else:
            runtime_error(self.lineno, "Variable '%s' is undefined in scope '%s'" % 
                (self.value, environment.scope))

    def __repr__(self):
        return repr(self.__dict__)


class number_token_value:
    def __init__(self, value, type, lineno):
        self.value = int(value)
        self.type = type
        self.lineno = lineno

    def eval(self, environment):
        return variable('INTEGER', self.value)

    def __repr__(self):
        return repr(self.__dict__)


def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.value = t.value.lower()
    if t.value in reserved_list:
        t.type = t.value.upper()
        t.value = t.type
    else:
        t.value = name_token_value(t.value, t.type, t.lineno)

    return t


def t_NUMBER(t):
    r'\d+'
    t.value = number_token_value(t.value, 'INTEGER', t.lineno)
    return t


# Ignored characters
t_ignore = " \t"


def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
import ply.lex as lex
lex.lex()


# Precedence rules for the arithmetic operators
precedence = (
    ('right', 'UNARYNOT'),
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'EQ', 'NE'),
    ('left', 'GT', 'GE', 'LT', 'LE'),
    ('left', 'ADD', 'SUB'),
    ('left', 'MUL', 'DIV', 'MOD'),
    ('right', 'UNARYSUB'),
)


# root of a syntax tree (of type 'block')
root = None


def runtime_error(lineno, message):
    from sys import exit

    print "Run-time error at line %d: %s" % (lineno, message)
    exit(1)


def parsetime_error(lineno, message):
    from sys import exit

    print "Parse-time error at line %d: %s" % (lineno, message)
    

def unreachable():
    raise Exception("Execution of unreachable code")


class token:
    def __init__(self, type, value):
        self.type = type
        self.value = value


class tokenizer:
    def __init__(self, string):
        self.tokens = string.split()
        self.pos = 0

    def next_token(self, lineno):
        if self.pos < len(self.tokens):
            value = self.tokens[self.pos].lower()
            self.pos += 1
            if value in ['false', 'true']:
                return token("BOOLEAN", eval(value.capitalize()))
            else:
                try:
                    int_value = int(value)
                    return token("INTEGER", int_value)
                except:
                    runtime_error(lineno, "Invalid token in input: '%s' is neither integer nor boolean literal" % (value,))
        else:
            runtime_error(lineno, "Unexpected end of input")


class environment:
    def __init__(self, variables=None, procedures=None, functions=None):
        if variables is None:
            variables = {}
        if procedures is None:
            procedures = {}
        if functions is None:
            functions = {}
        self.variables = variables
        self.procedures = procedures
        self.functions = functions      


empty_environment = environment()


class empty_action:
    def run(self, environment):
        pass


class actions:
    def __init__(self, actions=None):
        if actions is None:
            actions = []
        self.actions = actions

    def append(self, action):
        self.actions.append(action)

    def run(self, environment):
        for action in self.actions:
            action.run(environment)


class block:
    def __init__(self, environment, actions, scope):
        self.environment = environment
        self.actions = actions
        self.scope = scope
    
    def run(self):
        self.actions.run(self.environment)


class variable:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return repr(self.__dict__)


class array:
    def __init__(self, num_elements, elem_type):
        from copy import deepcopy
        self.type = 'ARRAY'
        self.storage = [deepcopy(elem_type) for i in range(num_elements)]        


class array_lookup:
    def __init__(self, name, indices):
        self.name = name
        self.lineno = name.lineno
        self.indices = indices
        self.internal_request = False

    def eval(self, environment):
        current_indices = [i.eval(environment) for i in self.indices]
        try:
            array = environment.variables[self.name.value]
        except LookupError:
            runtime_error(self.lineno, "Undefined variable '%s' in scope '%s'" % (self.name.value, environment.scope))
        index = 0
        self.lookup_repr = self.name.value + ''.join(['[%s]' % int(i.value) for i in current_indices])
        if not reduce(bool.__and__, [i.type == 'INTEGER' for i in current_indices]):
            runtime_error(self.lineno, "Invalid type of index in array lookup '%s'" %
                self.lookup_repr)
        current_indices = [i.value for i in current_indices]
        while index < len(current_indices):
            if array.type == 'ARRAY':
                if 0 <= current_indices[index] < len(array.storage):
                    array = array.storage[current_indices[index]]
                    index += 1
                else:
                    runtime_error(self.lineno, "Array index out of bounds (attempt to lookup '%s')" %
                        self.lookup_repr)
            else:
                runtime_error(self.lineno, "Invalid arity of array lookup '%s'" %
                    self.lookup_repr)
        scalar = array
        if scalar.value is None and not self.internal_request:
            runtime_error(self.lineno, "Attempt to use value of uninitialized variable '%s'" % (self.lookup_repr))
        if scalar.type not in SCALAR_TYPES:
            runtime_error(self.lineno, "Not enough indices in array lookup '%s'" % (self.lookup_repr))
        return scalar

    def set(self, environment, new_value):
        self.internal_request = True
        scalar = self.eval(environment)
        self.internal_request = False
        if scalar.type == new_value.type:
            scalar.value = new_value.value
        else:
            runtime_error(self.lineno,
                "Type mismatch in assignment to '%s': value of type %s expected, value of %s found" %
                (self.lookup_repr, scalar.type, new_value.type))


class assignment:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def run(self, environment):
        self.lhs.set(environment, self.rhs.eval(environment))


class read:
    def __init__(self, lhs):
        self.lhs = lhs

    def run(self, environment):
        token = input_tokenizer.next_token(self.lhs.lineno)
        self.lhs.set(environment, token)


class write:
    def __init__(self, expr):
        self.expr = expr

    def run(self, environment):
        print self.expr.eval(environment).value,


class writeln:
    def __init__(self, expr=None):
        self.expr = expr

    def run(self, environment):
        if self.expr is None:
            print
        else:
            print self.expr.eval(environment).value


class if_then_else:
    def __init__(self, condition, then_action, else_action):
        self.condition = condition
        self.then_action = then_action
        self.else_action = else_action

    def run(self, environment):
        condition = self.condition.eval(environment)
        if condition.type != 'BOOLEAN':
            runtime_error(self.condition.lineno, "Illegal type of condition in if-statement: BOOLEAN expected, %s found" % 
                (condition.type))
        if condition.value == True:
            self.then_action.run(environment)
        else:
            self.else_action.run(environment)


dummy_else = empty_action()


class while_do:
    def __init__(self, condition, actions):
        self.condition = condition
        self.actions = actions

    def run(self, environment):
        while True:
            condition = self.condition.eval(environment)
            if condition.type != 'BOOLEAN':
                runtime_error(self.condition.lineno, "Illegal type of condition in while-statement: BOOLEAN expected, %s found" %
                    (condition.type))
            if condition.value == True:
                self.actions.run(environment)
            else:
                break


class binop:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
        self.lineno = left.lineno

    def eval(self, environment):
        def type_mismatch():
            runtime_error(self.left.lineno, "Operator '%s' not applicable to operands of type '%s' and '%s'" % 
                (self.op, left.type, right.type))
            
        left = self.left.eval(environment)
        right = self.right.eval(environment)
        value = None
        type = None
        try:
            import operator
            if self.op in ('+', '-', '*', '>', '>=', '<', '<=', 'DIV', 'MOD'):
                if not (left.type == right.type == 'INTEGER'):
                    type_mismatch()
                value = {'+'  : operator.add,
                         '-'  : operator.sub,
                         '*'  : operator.mul,
                         '>'  : operator.gt,
                         '>=' : operator.ge,
                         '<'  : operator.lt,
                         '<=' : operator.le,
                         'DIV': operator.floordiv,
                         'MOD': operator.mod
                        }[self.op](left.value, right.value)
            if self.op in ('AND', 'OR'):
                if not (left.type == right.type == 'BOOLEAN'):
                    type_mismatch()
                value = {'AND': operator.and_,
                         'OR' : operator.or_
                        }[self.op](left.value, right.value)
            if self.op in ('=', '<>'):
                if not (left.type == right.type):
                    type_mismatch()
                value = {'=': operator.eq,
                         '<>': operator.ne
                        }[self.op](left.value, right.value)
            if self.op in ('+', '-', '*', 'DIV', 'MOD'):
                type = 'INTEGER'
            else:
                type = 'BOOLEAN'
            return variable(type, value)
        except ZeroDivisionError:
            runtime_error(self.left.lineno, "Division by zero ('%s' = %d, '%s' = %d)" %
                (self.left.value, left.value, self.right.value, right.value))


class unop:
    def __init__(self, op, right):
        self.op = op
        self.right = right
        self.lineno = right.lineno

    def eval(self, environment):
        def type_mismatch():
            runtime_error(self.left.lineno, "Operator '%s' not applicable to operand of type '%s'" %
                (self.op, right.type))

        right = self.right.eval(environment)
        value = None
        type = right.type
        if self.op == 'NOT':
            if right.type != 'BOOLEAN':
                type_mismatch()
            value = not right.value
        if self.op == '-':
            if right.type != 'INTEGER':
                type_mismatch()
            value = -right.value
        return variable(type, value)


def p_module(p):
    'module : MODULE NAME SEP declarations BEGIN actions END NAME FULLSTOP'
    global root
    if p[2].value != p[8].value:
        parsetime_error(p[8].lineno, "Illegal module name after END: '%s' expected, '%s' found" %
            (p[2].value, p[8].value))
    p[4].scope = 'module ' + p[2].value
    root = block(p[4], p[6], 'module ' + p[2].value)


def p_declarations(p):
    '''declarations : VAR varlist
                    | '''
    if len(p) == 3:
        p[0] = environment(variables=p[2])
    else:
        p[0] = environment()


def p_varlist_middle(p):
    'varlist : varlist varentry'
    p[1].update(p[2])
    p[0] = p[1]


def p_varlist_end(p):
    'varlist : varentry'
    p[0] = p[1]


def p_varentry(p):
    '''varentry : namelist COLON type SEP'''
    from copy import deepcopy

    type = None
    p[0] = {}
    # TODO: identifier redeclaration checking
    #       post-parsing dfs is right place to do that
    if p[3] in SCALAR_TYPES:
        type = variable(p[3], None)
    else:
        type = p[3]
    for name in p[1]:
        p[0][name.value] = deepcopy(type)


def p_type(p):
    '''type : BOOLEAN
            | INTEGER
            | ARRAY expression OF type'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        num_elements = p[2].eval(empty_environment)
        if num_elements.type != 'INTEGER':
            runtime_error(p.lineno(2), "Invalid type of array bound: INTEGER expected, %s found" %
                num_elements.type)
        num_elements = num_elements.value

        type = None
        if isinstance(p[4], str):
            type = variable(p[4], None)
        else:
            type = p[4]
            
        p[0] = array(num_elements, type)  


def p_namelist_middle(p):
    '''namelist : namelist COMMA NAME'''
    p[1].update([p[3]])
    p[0] = p[1]


def p_namelist_end(p):
    'namelist : NAME'
    p[0] = set([p[1]])


def p_actions(p):
    '''actions : actions action SEP
               | '''
    if len(p) == 1:
        p[0] = actions()
    else:
        p[1].append(p[2])
        p[0] = p[1]


def p_lvalue_name(p):
    'lvalue : NAME'
    p[0] = p[1]


def p_lvalue_array_lookup(p):
    'lvalue : array_lookup'
    p[0] = p[1]


def p_action_assignment(p):
    'action : lvalue ASSIGN expression'
    p[0] = assignment(p[1], p[3])


def p_action_read(p):
    'action : READ LPAREN lvalue RPAREN' 
    p[0] = read(p[3])


def p_action_write(p):
    'action : WRITE LPAREN expression RPAREN'
    p[0] = write(p[3])


def p_action_writeln(p):
    '''action : WRITELN LPAREN expression RPAREN
              | WRITELN LPAREN RPAREN '''
    if len(p) == 4:
        p[0] = writeln()
    else:
        p[0] = writeln(p[3])
 

def p_action_if_then(p):
    'action : IF expression THEN action'
    p[0] = if_then_else(p[2], p[4], dummy_else)


def p_action_if_then_else(p):
    'action : IF expression THEN action ELSE action'
    p[0] = if_then_else(p[2], p[4], p[6])


def p_action_while_do(p):
    'action : WHILE expression DO actions END'
    p[0] = while_do(p[2], p[4])


def p_action_begin_end(p):
    'action : BEGIN actions END'
    p[0] = p[2]


def p_expression_binop(p):
    '''expression : expression ADD expression
                  | expression SUB expression
                  | expression MUL expression
                  | expression DIV expression
                  | expression MOD expression
                  | expression EQ expression
                  | expression NE expression
                  | expression GT expression
                  | expression GE expression
                  | expression LT expression
                  | expression LE expression
                  | expression AND expression
                  | expression OR expression'''
    p[0] = binop(p[1], p[2], p[3])
        

def p_expression_unary_not(p):
    '''expression : NOT expression %prec UNARYNOT
                  | SUB expression %prec UNARYSUB'''
    p[0] = unop(p[1], p[2])


def p_expression_group(p):
    'expression : LPAREN expression RPAREN'
    p[0] = p[2]


def p_expression_number(p):
    'expression : NUMBER'
    p[0] = p[1]


def p_expression_name(p):
    'expression : NAME'
    p[0] = p[1]


def p_expression_array_lookup(p):
    'expression : array_lookup'
    p[0] = p[1]


def p_array_lookup(p):
    'array_lookup : NAME indices'
    p[0] = array_lookup(p[1], p[2])


def p_indices(p):
    '''indices : LBRA expression RBRA
               | LBRA expression RBRA indices'''
    if len(p) == 4:
        p[0] = [p[2]]
    else:
        p[0] = [p[2]] + p[4]


def p_expression_bool_literal(p):
    '''expression : TRUE
                | FALSE'''
    p[0] = p[1]
    p[0].value = eval(p[0].value.lower().capitalize())
    p[0].type = 'BOOLEAN'


def p_error(p):
    print("Syntax error at '%s', line %d" % (p.value, p.lineno))


import ply.yacc as yacc
yacc.yacc(start='module')


from sys import argv
s = open(argv[1], 'r').read()
yacc.parse(s, debug=int(argv[2]) if len(argv) > 2 else 0)


# Tokenizer for command 'read'

input_tokenizer = tokenizer(open('input.txt', 'r').read())

if root is not None:
    root.run()
else:
    print("Parsing failed")
