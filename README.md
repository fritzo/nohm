[![Build Status](https://github.com/fritzo/nohm/workflows/CI/badge.svg)](https://github.com/fritzo/nohm/actions)
![Maturity](https://img.shields.io/badge/maturity-prototype-red)

# NOHM: Nondeterministic Optimal Higher-order Machine

This is a research implementation of optimal beta reduction
([Asperti98a](#Asperti98a)), combining implementation ideas of BOHM
([Asperti98b](#Asperti98b)) with engineering tricks of HVM
([Taelin22](#Taelin22)).

The target language is pure untyped nondeterministic &lambda;-calculus
([Hindley08](#Hindley08)), that is the language with function abstraction,
function application, bound variables, and nondeterministic parallel binary
choice.
The NOHM aims to implement types as closures ([Scott76](#Scott76)) (increasing
idempotent functions) together with a rich type system of closures
([Obermeyer09](#Obermeyer09)).
It remains to be seen whether this is feasible.

The engineering plan is to create a readable and easily debuggable Python
runtime together with an equivalent but highly-optimized C runtime, similar to
HVM's hybrid Rust+C architecture ([Taelin22](#Taelin22)).
This architecture allows unit tests to be written in Python.

## References

<dl>
<dt> Asperti98a <a name="Asperti98a" /> </dt>
    <dd>
    Andrea Asperti, Stefano Guerrini (1998)
    "The optimal implementation of functional programming languages"
    </dd>
<dt> Asperti98b] <a name="Asperti98b" /> </dt>
    <dd>
    Andrea Asperti, Stefano Guerrini (1998)
    "Bolonga Optimal Higher-order Machine (BOHM)"
    https://github.com/asperti/BOHM1.1
    </dd>
<dt> Hindley08 <a name="Hindley08" /> </dt>
    <dd>
    J. Roger Hindley, J.P. Seldin (2008)
    "Lambda calculus and combinatory logic: an introduction"
    </dd>
<dt> Obermeyer09 <a name="Obermeyer09" /> </dt>
    <dd>
    Fritz Obermeyer (2009)
    "Automated equational reasoning in nondeterministic &lambda;-calculi modulo theories H*"
    http://fritzo.org/thesis.pdf
    </dd>
<dt> Salikhmetov17] <a name="Salikhmetov17" /> </dt>
    <dd>
    Anton Salikhmetov (2017)
    "inet-lib: JavaScript Engine for Interaction Nets"
    https://github.com/codedot/inet-lib
    </dd>
<dt> Scott76] <a name="Scott76" />
    <dd>
    Dana Scott (1976)
    "Datatypes as Lattices"
    http://www.cs.ox.ac.uk/files/3287/PRG05.pdf"
    </dd>
<dt> Taelin22] <a name="Taelin22" /> </dt>
    <dd>
    Victor Taelin et al. (2022)
    "Higher-order Virtual Machine (HVM)"
    https://github.com/Kindelia/HVM
    https://github.com/Kindelia/HVM/blob/master/HOW.md
    </dd>
</dl>

## License

Copyright (c) 2022 Fritz Obermeyer.<br/>
NOHM is licensed under the [Apache 2.0 License](/LICENSE).
