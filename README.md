[![Build Status](https://github.com/fritzo/nohm/workflows/CI/badge.svg)](https://github.com/fritzo/nohm/actions)
![Maturity](https://img.shields.io/badge/maturity-prototype-red)

# NOHM: Nondeterministic Optimal Higher-order Machine

This is a research implementation of optimal beta reduction
([Asperti98](#Asperti98)), combining implementation ideas of
[BOHM](https://github.com/asperti/BOHM1.1) ([Asperti96](#Asperti96)) with
engineering tricks of [HVM](https://github.com/Kindelia/HVM)
([Taelin22](#Taelin22)).

The target language is pure untyped nondeterministic &lambda;-calculus
([Barendregt84](#Barendregt84)), that is the language with function abstraction,
function application, bound variables, and nondeterministic parallel binary
choice.
This machine aims to implement types as closures ([Scott76](#Scott76))
(increasing idempotent functions) together with a rich type system of closures
([Obermeyer09](#Obermeyer09)).
It remains to be seen whether this is feasible.

The engineering plan is to create a readable and easily debuggable Python
runtime together with an equivalent but highly-optimized C runtime, similar to
HVM's hybrid Rust+C architecture ([Taelin22](#Taelin22)).
This architecture allows unit tests to be written in Python.

## References

<dl>
<dt> Asperti96 <a name="Asperti96" /> </dt>
    <dd>
    Andrea Asperti, Cecilia Giovanetti, Andrea Naletto (1996)
    "The Bologna Optimal Higher-order Machine"
    (<a href="https://dx.doi.org/10.1017/S0956796800001994">doi</a> |
    <a href="https://github.com/asperti/BOHM1.1">code</a>)
    </dd>
<dt> Asperti98 <a name="Asperti98" /> </dt>
    <dd>
    Andrea Asperti, Stefano Guerrini (1998)
    "The optimal implementation of functional programming languages"
    (<a href="https://doi.org/10.1145/505863.505887">doi</a>)
    </dd>
<dt>Barendregt84 <a name="Barendregt84" /> </dt>
    <dd>
    Hendrik Barendregt (1984)
    "The lambda calculus: its syntax and semantics"
<dt> Obermeyer09 <a name="Obermeyer09" /> </dt>
    <dd>
    Fritz Obermeyer (2009)
    "Automated equational reasoning in nondeterministic &lambda;-calculi modulo theories H*"
    (<a href="http://fritzo.org/thesis.pdf">pdf</a> |
    <a href="https://github.com/fritzo/johann">old code</a> |
    <a href="https://github.com/fritzo/pomagma">new code</a>)
    </dd>
<dt> Salikhmetov17 <a name="Salikhmetov17" /> </dt>
    <dd>
    Anton Salikhmetov (2017)
    "inet-lib: JavaScript Engine for Interaction Nets"
    (<a href="https://github.com/codedot/inet-lib">code</a> |
    <a href="https://arxiv.org/abs/1702.06092">paper</a>)
    </dd>
<dt> Scott76 <a name="Scott76" />
    <dd>
    Dana Scott (1976)
    "Datatypes as Lattices"
    (<a href="https://doi.org/10.1137/0205037">doi</a> |
     <a href="http://www.cs.ox.ac.uk/files/3287/PRG05.pdf">pdf</a>)
    </dd>
<dt> Taelin22 <a name="Taelin22" /> </dt>
    <dd>
    Victor Taelin et al. (2022)
    "Higher-order Virtual Machine (HVM)"
    (<a href="https://github.com/Kindelia/HVM">code</a> |
    <a href="https://github.com/Kindelia/HVM/blob/master/HOW.md">docs</a>)
    </dd>
</dl>

## License

Copyright (c) 2022 Fritz Obermeyer.<br/>
NOHM is licensed under the [Apache 2.0 License](/LICENSE).
