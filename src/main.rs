use core::fmt::{self, Display};
use std::{
    collections::{HashMap, VecDeque},
    iter::repeat,
};

use lang_c::{
    ast::{
        ArrayDeclarator, ArraySize, BinaryOperator, BinaryOperatorExpression, BlockItem, Constant,
        Declaration, DeclarationSpecifier, Declarator, DeclaratorKind, DerivedDeclarator,
        Expression, ExternalDeclaration, Identifier, InitDeclarator, Statement, TypeSpecifier,
        UnaryOperator, UnaryOperatorExpression,
    },
    driver::{parse, Config, Parse},
    span::Node,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Size {
    Int,
    Long,
}

impl Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Size::Int => writeln!(f, "    .long 0x0"),
            Size::Long => writeln!(f, "    .long 0x0"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Array {
    size: Size,
    length: usize,
}

impl Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        repeat(())
            .take(self.length)
            .map(|_| self.size.fmt(f))
            .collect::<fmt::Result>()
            .and_then(|_| writeln!(f, ""))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GlobalData {
    Value(Size),
    Array(Array),
    Pointer,
}

impl Display for GlobalData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlobalData::Value(size) => size.fmt(f),
            GlobalData::Array(array) => array.fmt(f),
            GlobalData::Pointer => writeln!(f, "    .long 0x0"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Global {
    identifier: String,
    data: GlobalData,
}

impl Display for Global {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.identifier)?;
        self.data.fmt(f)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
enum Register {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
}

impl Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Register::Zero => write!(f, "r0"),
            Register::One => write!(f, "r1"),
            Register::Two => write!(f, "r2"),
            Register::Three => write!(f, "r3"),
            Register::Four => write!(f, "r4"),
            Register::Five => write!(f, "r5"),
            Register::Six => write!(f, "r6"),
            Register::Seven => write!(f, "r7"),
        }
    }
}

struct Offset {
    base: Register,
    offset: usize,
}

impl Display for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.offset != 0 {
            write!(f, "{:#x}({})", self.offset, self.base)
        } else {
            write!(f, "({})", self.base)
        }
    }
}

struct Indexed {
    base: Register,
    index: Register,
    size: usize,
}

impl Display for Indexed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.base, self.index, self.size)
    }
}

enum Immediate {
    Global(String),
    Value(usize),
}

impl Display for Immediate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Immediate::Global(s) => write!(f, "${}", s),
            Immediate::Value(n) => write!(f, "${:#x}", n),
        }
    }
}

struct LoadImmediate {
    source: Immediate,
    destination: Register,
}

impl Display for LoadImmediate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    ld {}, {}", self.source, self.destination)
    }
}

struct LoadOffset {
    source: Offset,
    destination: Register,
}

impl Display for LoadOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    ld {}, {}", self.source, self.destination)
    }
}

struct LoadIndexed {
    source: Indexed,
    destination: Register,
}

impl Display for LoadIndexed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    ld {}, {}", self.source, self.destination)
    }
}

struct StoreOffset {
    source: Register,
    destination: Offset,
}

impl Display for StoreOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    st {}, {}", self.source, self.destination)
    }
}

struct StoreIndexed {
    source: Register,
    destination: Indexed,
}

impl Display for StoreIndexed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    st {}, {}", self.source, self.destination)
    }
}

struct Halt;

impl Display for Halt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    halt")
    }
}

struct Nop;

impl Display for Nop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    nop")
    }
}

struct Move {
    source: Register,
    destination: Register,
}

impl Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    mov {}, {}", self.source, self.destination)
    }
}

struct Add {
    source: Register,
    destination: Register,
}

impl Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    add {}, {}", self.source, self.destination)
    }
}

struct And {
    source: Register,
    destination: Register,
}

impl Display for And {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    and {}, {}", self.source, self.destination)
    }
}

struct Increment {
    destination: Register,
}

impl Display for Increment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    inc {}", self.destination)
    }
}

struct IncrementAddress {
    destination: Register,
}

impl Display for IncrementAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    inca {}", self.destination)
    }
}

struct Decrement {
    destination: Register,
}

impl Display for Decrement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    dec {}", self.destination)
    }
}

struct DecrementAddress {
    destination: Register,
}

impl Display for DecrementAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    deca {}", self.destination)
    }
}

struct Not {
    destination: Register,
}

impl Display for Not {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    not {}", self.destination)
    }
}

struct ShiftLeft {
    offset: Immediate,
    destination: Register,
}

impl Display for ShiftLeft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    shl {}, {}", self.offset, self.destination)
    }
}

struct ShiftRight {
    offset: Immediate,
    destination: Register,
}

impl Display for ShiftRight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    shr {}, {}", self.offset, self.destination)
    }
}

fn determine_size(specifiers: &Vec<Node<DeclarationSpecifier>>) -> Size {
    let mut sizes = specifiers.iter().filter_map(|Node { node, .. }| {
        if let DeclarationSpecifier::TypeSpecifier(Node { node: spec, .. }) = node {
            match spec {
                TypeSpecifier::Int => Some(Size::Int),
                TypeSpecifier::Long => Some(Size::Long),
                _ => None,
            }
        } else {
            None
        }
    });

    let Some(size) = sizes.next() else {
        panic!("couldn't determine type for global declaration")
    };

    if let Some(_) = sizes.next() {
        panic!("more than one potential size for global declaration")
    }

    size
}

fn determine_array_length(arr_decl: &ArrayDeclarator) -> usize {
    let ArraySize::VariableExpression(ref expr) = arr_decl.size else {
        panic!()
    };

    let Expression::Constant(ref constant) = expr.node else {
        panic!()
    };

    let Constant::Integer(ref n) = constant.node else {
        panic!()
    };

    n.number.parse().expect("failed to parse array size")
}

fn create_global(decl: &Declarator, size: Size) -> Global {
    let DeclaratorKind::Identifier(Node{node: Identifier {ref name}, ..}) = decl.kind.node else {
        panic!("unsupported global")
    };

    for Node { node: derived, .. } in decl.derived.iter() {
        match derived {
            DerivedDeclarator::Pointer(_) => {
                return Global {
                    identifier: name.to_owned(),
                    data: GlobalData::Pointer,
                }
            }
            DerivedDeclarator::Array(Node { node: arr_decl, .. }) => {
                return Global {
                    identifier: name.to_owned(),
                    data: GlobalData::Array(Array {
                        size,
                        length: determine_array_length(arr_decl),
                    }),
                }
            }
            _ => (),
        }
    }

    Global {
        identifier: name.to_owned(),
        data: GlobalData::Value(size),
    }
}

fn create_globals(decls: &Vec<Node<InitDeclarator>>, size: Size) -> Vec<Global> {
    decls
        .iter()
        .map(|Node { node: decl, .. }| create_global(&decl.declarator.node, size))
        .collect()
}

fn compile_global(decl: &Declaration) -> Vec<Global> {
    create_globals(&decl.declarators, determine_size(&decl.specifiers))
}

fn compile_globals(parse: &Parse) -> Vec<Global> {
    let unit = &parse.unit.0;
    unit.iter()
        .flat_map(|Node { node, .. }| {
            if let ExternalDeclaration::Declaration(Node { node: decl, .. }) = node {
                compile_global(decl)
            } else {
                vec![]
            }
        })
        .collect()
}

fn get_main(parse: &Parse) -> &Statement {
    let unit = &parse.unit.0;

    for decl in unit {
        let ExternalDeclaration::FunctionDefinition(Node{node: ref def, ..}) = decl.node else {
            continue
        };

        let DeclaratorKind::Identifier(Node{node: Identifier { ref name }, ..})= def.declarator.node.kind.node else {
            panic!("function declaration doesn't have a name")
        };

        if name == "main" {
            return &def.statement.node;
        }
    }

    panic!("missing main function")
}

struct Context {
    globals: Vec<Global>,
    instructions: Vec<Box<dyn Display>>,
    reference_registers: HashMap<Global, Register>,
    available_registers: VecDeque<Register>,
    pending_deletion: VecDeque<Register>,
}

impl Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".pos 0x100")?;
        for instruction in self.instructions.iter() {
            instruction.fmt(f)?;
        }
        let mut offset = 0x1000usize;
        for global in self.globals.iter() {
            writeln!(f, ".pos {:#x}", offset)?;
            global.fmt(f)?;
            offset += match global.data {
                GlobalData::Value(_) => 4,                // TODO use size
                GlobalData::Array(arr) => arr.length * 4, // TODO use size
                GlobalData::Pointer => 4,
            }
        }
        Ok(())
    }
}

impl Context {
    fn get_global_reference(&mut self, name: &str) -> Result<Register, Option<Register>> {
        let Some(global_reference) = self
            .globals
            .iter()
            .find(|global| global.identifier == name) else {
                return Err(None)
            };
        let global = global_reference.to_owned();

        if let Some(register) = self.reference_registers.get(&global) {
            // println!("getting {} from {}", global.identifier, register);
            Ok(*register)
        } else {
            let Some(register) = self.get_unused_register() else {
                return Err(None)
            };
            // println!(
            //     "claiming {} as a register for {}",
            //     register, global.identifier
            // );
            self.reference_registers.insert(global, register);
            Err(Some(register))
        }
    }

    fn get_unused_register(&mut self) -> Option<Register> {
        if let Some(register) = self.available_registers.pop_front() {
            // println!("claiming unused register {}", register);
            Some(register)
        } else if let Some(register) = self.pending_deletion.pop_back() {
            // println!("reusing a deleted register {}", register);
            if let Some(entry) = self
                .reference_registers
                .iter()
                .find(|(_, &v)| v == register)
            {
                let global = entry.0.to_owned();
                // println!(
                //     "{} is not longer stored in this register",
                //     global.identifier
                // );
                self.reference_registers.remove(&global);
            }
            Some(register)
        } else {
            None
        }
    }

    fn done_with_register(&mut self, register: Register) {
        if !self.pending_deletion.contains(&register) {
            // println!("marked {} for reuse", register);
            self.pending_deletion.push_back(register);
        } else {
            // println!("{} is already marked for reuse", register);
        }
    }

    fn add_instruction(&mut self, changes: Option<Register>, instruction: impl Display + 'static) {
        if let Some(register) = changes {
            if let Some(entry) = self
                .reference_registers
                .iter()
                .find(|(_, &v)| v == register)
            {
                let global = entry.0.to_owned();
                // println!(
                //     "this instruction removes {} from {}",
                //     global.identifier, register
                // );
                self.reference_registers.remove(&global);
            }
        }

        // println!("{}\n", instruction);
        self.instructions.push(Box::new(instruction));
    }
}

fn parse_identifier_assignment(ident: &Identifier, rhs: Register, ctx: &mut Context) {
    let destination = parse_identifier_reference(ident, ctx);

    ctx.add_instruction(
        None,
        StoreOffset {
            source: rhs,
            destination: Offset {
                base: destination,
                offset: 0,
            },
        },
    );

    ctx.done_with_register(destination);
}

fn parse_indexed_assignment(binop: &BinaryOperatorExpression, rhs: Register, ctx: &mut Context) {
    let BinaryOperator::Index = binop.operator.node else {
        panic!("")
    };

    let Expression::Identifier(ident) = &binop.lhs.node else {
        panic!("")
    };

    if let Expression::Constant(cnst) = &binop.rhs.node {
        let Constant::Integer(num) = &cnst.node else {
            panic!("can only index by int")
        };

        let offset = num.number.parse::<usize>().unwrap() * 4;
        let base = parse_identifier_reference(&ident.node, ctx);

        ctx.add_instruction(
            None,
            StoreOffset {
                source: rhs,
                destination: Offset { base, offset },
            },
        );

        ctx.done_with_register(base);
    } else {
        let index = parse_expression(&binop.rhs.node, ctx).unwrap();
        let base = parse_identifier_reference(&ident.node, ctx);
        ctx.add_instruction(
            None,
            StoreIndexed {
                source: rhs,
                destination: Indexed {
                    base,
                    index,
                    size: 4, // TODO fix size
                },
            },
        );
        ctx.done_with_register(index);
        ctx.done_with_register(base);
    }
}

fn parse_pointer_assignment(unop: &UnaryOperatorExpression, rhs: Register, ctx: &mut Context) {
    let UnaryOperator::Indirection = unop.operator.node else {
        panic!("")
    };

    let Expression::Identifier(ident) = &unop.operand.node else {
        panic!("")
    };

    let dereferenced = parse_identifier(&ident.node, ctx);

    ctx.add_instruction(
        None,
        StoreOffset {
            source: rhs,
            destination: Offset {
                base: dereferenced,
                offset: 0,
            },
        },
    );

    ctx.done_with_register(dereferenced);
}

fn parse_assignment(expr: &Expression, rhs: Register, ctx: &mut Context) {
    match expr {
        Expression::Identifier(ref ident) => parse_identifier_assignment(&ident.node, rhs, ctx),
        Expression::UnaryOperator(ref unop) => parse_pointer_assignment(&unop.node, rhs, ctx),
        Expression::BinaryOperator(ref binop) => parse_indexed_assignment(&binop.node, rhs, ctx),
        _ => panic!("invalid left hand side of assignment"),
    }
}

fn parse_binary_operator(binop: &BinaryOperatorExpression, ctx: &mut Context) -> Option<Register> {
    let op = &binop.operator.node;

    if let BinaryOperator::Assign = op {
        let rhs = parse_expression(&binop.rhs.node, ctx).unwrap();
        parse_assignment(&binop.lhs.node, rhs, ctx);
        ctx.done_with_register(rhs);
        return None;
    }

    if let BinaryOperator::Index = op {
        let Expression::Identifier(ident) = &binop.lhs.node else {
            panic!("")
        };

        if let Expression::Constant(cnst) = &binop.rhs.node {
            let Constant::Integer(num) = &cnst.node else {
                panic!("can only index by int")
            };

            let offset = num.number.parse::<usize>().unwrap() * 4;
            let lhs = parse_identifier_reference(&ident.node, ctx);
            let destination = ctx.get_unused_register().expect("no registers left");

            ctx.add_instruction(
                Some(destination),
                LoadOffset {
                    source: Offset { base: lhs, offset },
                    destination,
                },
            );

            ctx.done_with_register(lhs);
            return Some(destination);
        } else {
            let rhs = parse_expression(&binop.rhs.node, ctx).unwrap();
            let lhs = parse_identifier_reference(&ident.node, ctx);
            let destination = ctx.get_unused_register().expect("no registers left");

            ctx.add_instruction(
                Some(destination),
                LoadIndexed {
                    source: Indexed {
                        base: lhs,
                        index: rhs,
                        size: 4, // TODO wrong
                    },
                    destination,
                },
            );

            ctx.done_with_register(lhs);
            ctx.done_with_register(rhs);
            return Some(destination);
        }
    }

    let rhs = parse_expression(&binop.rhs.node, ctx).unwrap();
    let lhs = parse_expression(&binop.lhs.node, ctx).unwrap();

    match op {
        BinaryOperator::Multiply => {
            ctx.add_instruction(
                Some(lhs),
                ShiftLeft {
                    offset: todo!(),
                    destination: lhs,
                },
            );
            ctx.done_with_register(rhs);
            Some(lhs)
        }
        BinaryOperator::Divide => {
            ctx.add_instruction(
                Some(lhs),
                ShiftRight {
                    offset: todo!(),
                    destination: lhs,
                },
            );
            ctx.done_with_register(rhs);
            Some(lhs)
        }
        BinaryOperator::Plus => {
            ctx.add_instruction(
                Some(lhs),
                Add {
                    source: rhs,
                    destination: lhs,
                },
            );
            ctx.done_with_register(rhs);
            Some(lhs)
        }
        BinaryOperator::Minus => todo!(),
        BinaryOperator::ShiftLeft => todo!(),
        BinaryOperator::ShiftRight => todo!(),
        BinaryOperator::BitwiseAnd => {
            ctx.add_instruction(
                Some(lhs),
                And {
                    source: rhs,
                    destination: lhs,
                },
            );
            ctx.done_with_register(rhs);
            Some(lhs)
        }
        _ => unimplemented!(),
    }
}

fn parse_constant(cnst: &Constant, ctx: &mut Context) -> Register {
    match cnst {
        Constant::Integer(ref num) => {
            let value = num.number.parse::<usize>().unwrap();
            let destination = ctx.get_unused_register().unwrap();
            ctx.add_instruction(
                Some(destination),
                LoadImmediate {
                    source: Immediate::Value(value),
                    destination,
                },
            );
            destination
        }
        _ => unimplemented!(),
    }
}

fn dereference(reference: Register, ctx: &mut Context) -> Register {
    let value = ctx.get_unused_register().unwrap();
    ctx.add_instruction(
        Some(value),
        LoadOffset {
            source: Offset {
                base: reference,
                offset: 0,
            },
            destination: value,
        },
    );
    value
}

fn parse_identifier(ident: &Identifier, ctx: &mut Context) -> Register {
    let reference = parse_identifier_reference(ident, ctx);
    let value = dereference(reference, ctx);
    ctx.done_with_register(reference);
    value
}

fn parse_identifier_reference(ident: &Identifier, ctx: &mut Context) -> Register {
    match ctx.get_global_reference(&ident.name) {
        Ok(reference) => reference,
        Err(Some(reference)) => {
            ctx.add_instruction(
                None, // Valid because we are setting the global's address
                LoadImmediate {
                    source: Immediate::Global(ident.name.to_owned()),
                    destination: reference,
                },
            );
            reference
        }
        Err(None) => panic!("ran out of registers"),
    }
}

fn parse_expression(expr: &Expression, ctx: &mut Context) -> Option<Register> {
    match expr {
        Expression::Identifier(ref ident) => Some(parse_identifier(&ident.node, ctx)),
        Expression::Constant(ref cnst) => Some(parse_constant(&cnst.node, ctx)),
        Expression::UnaryOperator(ref unop) => Some(parse_unary(&unop.node, ctx)),
        Expression::Cast(_) => todo!(),
        Expression::BinaryOperator(ref binop) => parse_binary_operator(&binop.node, ctx),
        _ => unimplemented!(),
    }
}

fn parse_unary(unop: &UnaryOperatorExpression, ctx: &mut Context) -> Register {
    let Expression::Identifier(ident) = &unop.operand.node else {
        panic!("")
    };
    let reference = parse_identifier_reference(&ident.node, ctx);
    match unop.operator.node {
        UnaryOperator::PostIncrement => {
            let value = dereference(reference, ctx);
            ctx.add_instruction(Some(value), Increment { destination: value });
            ctx.add_instruction(
                None,
                StoreOffset {
                    source: value,
                    destination: Offset {
                        base: reference,
                        offset: 0,
                    },
                },
            );
            ctx.done_with_register(reference);
            value
        }
        UnaryOperator::PostDecrement => {
            let value = dereference(reference, ctx);
            ctx.add_instruction(Some(value), Decrement { destination: value });
            ctx.add_instruction(
                None,
                StoreOffset {
                    source: value,
                    destination: Offset {
                        base: reference,
                        offset: 0,
                    },
                },
            );
            ctx.done_with_register(reference);
            value
        }
        _ => unimplemented!(),
    }
}

fn parse_statement(stmt: &Statement, ctx: &mut Context) -> Option<Register> {
    match stmt {
        Statement::Expression(Some(ref node)) => parse_expression(&node.node, ctx),
        Statement::Compound(nodes) => {
            nodes.iter().for_each(|node| match node.node {
                BlockItem::Statement(Node { node: ref stmt, .. }) => {
                    if let Some(destination) = parse_statement(stmt, ctx) {
                        ctx.done_with_register(destination);
                    }
                }
                _ => todo!(),
            });
            None
        }
        _ => unimplemented!(),
    }
}

fn main() {
    let config = Config::default();
    let parsed = parse(&config, "q2.c").unwrap();

    let globals = compile_globals(&parsed);

    let main = get_main(&parsed);

    let mut available_registers = VecDeque::new();
    available_registers.push_back(Register::Zero);
    available_registers.push_back(Register::One);
    available_registers.push_back(Register::Two);
    available_registers.push_back(Register::Three);
    available_registers.push_back(Register::Four);
    available_registers.push_back(Register::Five);
    available_registers.push_back(Register::Six);
    available_registers.push_back(Register::Seven);
    let mut ctx = Context {
        globals,
        instructions: vec![],
        reference_registers: HashMap::new(),
        available_registers,
        pending_deletion: VecDeque::new(),
    };

    parse_statement(main, &mut ctx);
    ctx.add_instruction(None, Halt);

    println!("{}", ctx);
}
