# Reinfier
A universal verification framework system for deep reinforcement learning, which combines the formal verification of deep neural network with bounded model checking algorithm and k-induction algorithm to verify the properties of deep reinforcement learning or give counterexamples.
## Installation
Reinfier requires python>=3.8. To install Reinfier, run:  
```shell
pip install reinfier
```
Reinfier is based on [DNNV](https://github.com/dlshriver/dnnv), which requrires verifiers of DNN ([Reluplex](https://github.com/guykatzz/ReluplexCav2017), [planet](https://github.com/progirep/planet), [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl), [Neurify](https://github.com/tcwangshiqi-columbia/Neurify), [ERAN](https://github.com/eth-sri/eran), [BaB](https://github.com/oval-group/PLNN-verification), [marabou](https://github.com/NeuralNetworkVerification/Marabou), [nnenum](https://github.com/stanleybak/nnenum), [verinet](https://vas.doc.ic.ac.uk/software/neural/)).  
To install any of the supported verifiers, run:
```shell
dnnv_manage install reluplex planet mipverify neurify eran bab marabou nnenum verinet
```

Usage sample files to test:  
```python
import reinfier as rf

dnn,drlp=rf.test.get_sample_path()

rf.k_induction(dnn,drlp)
```
The result should be:
```python
(2, False)
```
which means the property is False (SAT, Invalid) with verification depth is 2.

## Usage
```python
import reinfier as rf

dnn  = /path/to/onnx/deep/neural/network/file

drlp = /path/to/drlp/file
# or
drlp = drlp_str

rf.k_induction(dnn,drlp) # k-induction algorithm (Recommended)
# or
rf.bmc(dnn,drlp) # bounded model checking algorithm
```

## DRLP
DRLP, i.e. Deep Reinforcement Learning Property, is a Pyhton-embedded DSL to describe property of DRL.
### Reserved Keywords
| Parameter                | Variable Keyword |       Type      |
|--------------------------|:----------------:|:---------------:|
| Input of NN $x$          |        $x$       | $numpy.ndarray$ |
| Output of NN $y$         |        $y$       | $numpy.ndarray$ |
| Input size of NN         |     $x\_size$    |      $int$      |
| Output size of NN        |     $y\_size$    |      $int$      |
| Verification depths $k$  |        $k$       |      $int$      |
### Example
```python
[[-1]*2]*k <= x <= [[1]*2]*k

[0]*2 <= x[0] <= [0]*2

for i in range(0,k-1):
    Implies(y[i] > [0],  x[i]+0.5 >= x[i+1] >= x[i])
    Implies(y[i] <= [0], x[i]-0.5 <= x[i+1] <= x[i])

# Exp
y >= [[-2]]*k
```
Such DRLP text describe an Environment and an Agent:  
1. State boundary 𝑆: Each input value is within $[−1,1]$  
2. Initial state 𝐼: Each input value is $0$  
3. State transition 𝑇: Each input value of the next state increases by at most $0.5$ when output is greater than $0$, and each input value of the next state decreases by at most $0.5$ when output is not greater than $0$
4. Other constraints 𝐶: None
5. Property 𝑄: Output is always not less than $-2$ 

### Defination

The dinfination of DRLP:  
```BNF
<drlp> ::= <io_size_assign> NEWLINE <statements> NEWLINE  '# Exp' NEWLINE <statements>

<io_size_assign> ::= ''
   |  <io_size_assign> NEWLINE <io_size_id> '=' <int>
   
<io_size_id> = 'x_size' | 'y_size'

<statements> ::= ''
    | <statements> NEWLINE <statement>

<statement> ::= <compound_stmt> | <simple_stmts>

<compound_stmt> ::= <for_stmt> | <with_stmt>

<for_stmt> :: = 'for' <id> 'in' <range_type> <for_range> ':' <block>

<with_stmt> :: = 'with'  <range_type> ':' <block>

<block> ::= NEWLINE INDENT <statements> DEDENT
    | <simple_stmts>

<range_type> ::= 'range' | 'orange'

<for_range> ::= '('<int>')'
    | '('<int> ',' <int> ')'
    | '('<int> ',' <int> ',' <int>')'

<simple_stmts> ::= ""
    | <simple_stmts> NEWLINE <simple_stmt>

<simple_stmt> ::= <call_stmt> | <expr>

<call_stmt> ::= 'Impiles' '(' <expr> ',' <expr> ')'
    | 'And' '(' <exprs> ')'
    | 'Or' '(' <exprs> ')'

<exprs> ::= <expr> 
    | <exprs> ',' <expr>

<expr> ::= <obj> <comparation>

<comparation> ::= '' 
    | <comparator> <obj> <comparation>

<obj> ::= <constraint> | <io_obj>

<io_obj> ::= <io_id> 
    | <io_id> <subscript>
    | <io_id> <subscript> <subscript>
    
<io_id> ::= 'x' | 'y'

<subscript> ::= '[' <int> ']'
     | '[' <int>':'<int> ']'
     | '[' <int>':'<int> ':'<int>']'

<int> ::= <int_number> 
    | <id> 
    | <int> <operator> <int>

<constraint> :: = <int> 
    | <list>
    | <constraint> <operator> <constraint>

```