э╣
У+Ў*
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
>
DiagPart

input"T
diagonal"T"
Ttype:

2	
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
н
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Q
Qr

input"T
q"T
r"T"
full_matricesbool( "
Ttype:	
2
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
е
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sign
x"T
y"T"
Ttype:

2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:И
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeИ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
TtypeИ
9
TensorArraySizeV3

handle
flow_in
sizeИ
▐
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring И
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
TtypeИ
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.15.02unknown8ж╟


global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
shape: *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Й
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
p
PlaceholderPlaceholder*
shape:         Ї*
dtype0*(
_output_shapes
:         Ї
п
5embedding/embeddings/Initializer/random_uniform/shapeConst*
valueB"jЭ  ╚   *'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
:
б
3embedding/embeddings/Initializer/random_uniform/minConst*
valueB
 *═╠L╜*'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
: 
б
3embedding/embeddings/Initializer/random_uniform/maxConst*
valueB
 *═╠L=*'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
: 
ч
=embedding/embeddings/Initializer/random_uniform/RandomUniformRandomUniform5embedding/embeddings/Initializer/random_uniform/shape*
T0*'
_class
loc:@embedding/embeddings*
dtype0*!
_output_shapes
:ъ║╚
ю
3embedding/embeddings/Initializer/random_uniform/subSub3embedding/embeddings/Initializer/random_uniform/max3embedding/embeddings/Initializer/random_uniform/min*
T0*'
_class
loc:@embedding/embeddings*
_output_shapes
: 
Г
3embedding/embeddings/Initializer/random_uniform/mulMul=embedding/embeddings/Initializer/random_uniform/RandomUniform3embedding/embeddings/Initializer/random_uniform/sub*
T0*'
_class
loc:@embedding/embeddings*!
_output_shapes
:ъ║╚
ї
/embedding/embeddings/Initializer/random_uniformAdd3embedding/embeddings/Initializer/random_uniform/mul3embedding/embeddings/Initializer/random_uniform/min*
T0*'
_class
loc:@embedding/embeddings*!
_output_shapes
:ъ║╚
░
embedding/embeddingsVarHandleOp*
shape:ъ║╚*%
shared_nameembedding/embeddings*'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
: 
y
5embedding/embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOpembedding/embeddings*
_output_shapes
: 
Г
embedding/embeddings/AssignAssignVariableOpembedding/embeddings/embedding/embeddings/Initializer/random_uniform*
dtype0
А
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
dtype0*!
_output_shapes
:ъ║╚
e
embedding/CastCastPlaceholder*

SrcT0*(
_output_shapes
:         Ї*

DstT0
╟
embedding/embedding_lookupResourceGatherembedding/embeddingsembedding/Cast*
Tindices0*'
_class
loc:@embedding/embeddings*
dtype0*-
_output_shapes
:         Ї╚
м
#embedding/embedding_lookup/IdentityIdentityembedding/embedding_lookup*
T0*'
_class
loc:@embedding/embeddings*-
_output_shapes
:         Ї╚
О
%embedding/embedding_lookup/Identity_1Identity#embedding/embedding_lookup/Identity*
T0*-
_output_shapes
:         Ї╚
╟
Abidirectional/forward_gru/kernel/Initializer/random_uniform/shapeConst*
valueB"╚   `   *3
_class)
'%loc:@bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
:
╣
?bidirectional/forward_gru/kernel/Initializer/random_uniform/minConst*
valueB
 *i╩╛*3
_class)
'%loc:@bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
: 
╣
?bidirectional/forward_gru/kernel/Initializer/random_uniform/maxConst*
valueB
 *i╩>*3
_class)
'%loc:@bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
: 
Й
Ibidirectional/forward_gru/kernel/Initializer/random_uniform/RandomUniformRandomUniformAbidirectional/forward_gru/kernel/Initializer/random_uniform/shape*
T0*3
_class)
'%loc:@bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
:	╚`
Ю
?bidirectional/forward_gru/kernel/Initializer/random_uniform/subSub?bidirectional/forward_gru/kernel/Initializer/random_uniform/max?bidirectional/forward_gru/kernel/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@bidirectional/forward_gru/kernel*
_output_shapes
: 
▒
?bidirectional/forward_gru/kernel/Initializer/random_uniform/mulMulIbidirectional/forward_gru/kernel/Initializer/random_uniform/RandomUniform?bidirectional/forward_gru/kernel/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@bidirectional/forward_gru/kernel*
_output_shapes
:	╚`
г
;bidirectional/forward_gru/kernel/Initializer/random_uniformAdd?bidirectional/forward_gru/kernel/Initializer/random_uniform/mul?bidirectional/forward_gru/kernel/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@bidirectional/forward_gru/kernel*
_output_shapes
:	╚`
╥
 bidirectional/forward_gru/kernelVarHandleOp*
shape:	╚`*1
shared_name" bidirectional/forward_gru/kernel*3
_class)
'%loc:@bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
: 
С
Abidirectional/forward_gru/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp bidirectional/forward_gru/kernel*
_output_shapes
: 
з
'bidirectional/forward_gru/kernel/AssignAssignVariableOp bidirectional/forward_gru/kernel;bidirectional/forward_gru/kernel/Initializer/random_uniform*
dtype0
Ц
4bidirectional/forward_gru/kernel/Read/ReadVariableOpReadVariableOp bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
:	╚`
┌
Jbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/shapeConst*
valueB"`       *=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes
:
═
Ibidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/meanConst*
valueB
 *    *=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
╧
Kbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/stddevConst*
valueB
 *  А?*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
▓
Ybidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalJbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/shape*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes

:` 
▀
Hbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/mulMulYbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/RandomStandardNormalKbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/stddev*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes

:` 
╚
Dbidirectional/forward_gru/recurrent_kernel/Initializer/random_normalAddHbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/mulIbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal/mean*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes

:` 
ў
9bidirectional/forward_gru/recurrent_kernel/Initializer/QrQrDbidirectional/forward_gru/recurrent_kernel/Initializer/random_normal*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*(
_output_shapes
:` :  
ь
?bidirectional/forward_gru/recurrent_kernel/Initializer/DiagPartDiagPart;bidirectional/forward_gru/recurrent_kernel/Initializer/Qr:1*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes
: 
ш
;bidirectional/forward_gru/recurrent_kernel/Initializer/SignSign?bidirectional/forward_gru/recurrent_kernel/Initializer/DiagPart*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes
: 
б
:bidirectional/forward_gru/recurrent_kernel/Initializer/mulMul9bidirectional/forward_gru/recurrent_kernel/Initializer/Qr;bidirectional/forward_gru/recurrent_kernel/Initializer/Sign*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes

:` 
ц
Vbidirectional/forward_gru/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*
valueB"       *=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes
:
┌
Qbidirectional/forward_gru/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose:bidirectional/forward_gru/recurrent_kernel/Initializer/mulVbidirectional/forward_gru/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes

: `
╘
Dbidirectional/forward_gru/recurrent_kernel/Initializer/Reshape/shapeConst*
valueB"    `   *=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes
:
╩
>bidirectional/forward_gru/recurrent_kernel/Initializer/ReshapeReshapeQbidirectional/forward_gru/recurrent_kernel/Initializer/matrix_transpose/transposeDbidirectional/forward_gru/recurrent_kernel/Initializer/Reshape/shape*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes

: `
┬
>bidirectional/forward_gru/recurrent_kernel/Initializer/mul_1/xConst*
valueB
 *  А?*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
л
<bidirectional/forward_gru/recurrent_kernel/Initializer/mul_1Mul>bidirectional/forward_gru/recurrent_kernel/Initializer/mul_1/x>bidirectional/forward_gru/recurrent_kernel/Initializer/Reshape*
T0*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
_output_shapes

: `
я
*bidirectional/forward_gru/recurrent_kernelVarHandleOp*
shape
: `*;
shared_name,*bidirectional/forward_gru/recurrent_kernel*=
_class3
1/loc:@bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
е
Kbidirectional/forward_gru/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp*bidirectional/forward_gru/recurrent_kernel*
_output_shapes
: 
╝
1bidirectional/forward_gru/recurrent_kernel/AssignAssignVariableOp*bidirectional/forward_gru/recurrent_kernel<bidirectional/forward_gru/recurrent_kernel/Initializer/mul_1*
dtype0
й
>bidirectional/forward_gru/recurrent_kernel/Read/ReadVariableOpReadVariableOp*bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
░
0bidirectional/forward_gru/bias/Initializer/zerosConst*
valueB`*    *1
_class'
%#loc:@bidirectional/forward_gru/bias*
dtype0*
_output_shapes
:`
╟
bidirectional/forward_gru/biasVarHandleOp*
shape:`*/
shared_name bidirectional/forward_gru/bias*1
_class'
%#loc:@bidirectional/forward_gru/bias*
dtype0*
_output_shapes
: 
Н
?bidirectional/forward_gru/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpbidirectional/forward_gru/bias*
_output_shapes
: 
Ш
%bidirectional/forward_gru/bias/AssignAssignVariableOpbidirectional/forward_gru/bias0bidirectional/forward_gru/bias/Initializer/zeros*
dtype0
Н
2bidirectional/forward_gru/bias/Read/ReadVariableOpReadVariableOpbidirectional/forward_gru/bias*
dtype0*
_output_shapes
:`
╔
Bbidirectional/backward_gru/kernel/Initializer/random_uniform/shapeConst*
valueB"╚   `   *4
_class*
(&loc:@bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
:
╗
@bidirectional/backward_gru/kernel/Initializer/random_uniform/minConst*
valueB
 *i╩╛*4
_class*
(&loc:@bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
: 
╗
@bidirectional/backward_gru/kernel/Initializer/random_uniform/maxConst*
valueB
 *i╩>*4
_class*
(&loc:@bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
: 
М
Jbidirectional/backward_gru/kernel/Initializer/random_uniform/RandomUniformRandomUniformBbidirectional/backward_gru/kernel/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
:	╚`
в
@bidirectional/backward_gru/kernel/Initializer/random_uniform/subSub@bidirectional/backward_gru/kernel/Initializer/random_uniform/max@bidirectional/backward_gru/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@bidirectional/backward_gru/kernel*
_output_shapes
: 
╡
@bidirectional/backward_gru/kernel/Initializer/random_uniform/mulMulJbidirectional/backward_gru/kernel/Initializer/random_uniform/RandomUniform@bidirectional/backward_gru/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@bidirectional/backward_gru/kernel*
_output_shapes
:	╚`
з
<bidirectional/backward_gru/kernel/Initializer/random_uniformAdd@bidirectional/backward_gru/kernel/Initializer/random_uniform/mul@bidirectional/backward_gru/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@bidirectional/backward_gru/kernel*
_output_shapes
:	╚`
╒
!bidirectional/backward_gru/kernelVarHandleOp*
shape:	╚`*2
shared_name#!bidirectional/backward_gru/kernel*4
_class*
(&loc:@bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
: 
У
Bbidirectional/backward_gru/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!bidirectional/backward_gru/kernel*
_output_shapes
: 
к
(bidirectional/backward_gru/kernel/AssignAssignVariableOp!bidirectional/backward_gru/kernel<bidirectional/backward_gru/kernel/Initializer/random_uniform*
dtype0
Ш
5bidirectional/backward_gru/kernel/Read/ReadVariableOpReadVariableOp!bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
:	╚`
▄
Kbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/shapeConst*
valueB"`       *>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes
:
╧
Jbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/meanConst*
valueB
 *    *>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
╤
Lbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/stddevConst*
valueB
 *  А?*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
╡
Zbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalKbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/shape*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes

:` 
у
Ibidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/mulMulZbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/RandomStandardNormalLbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/stddev*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes

:` 
╠
Ebidirectional/backward_gru/recurrent_kernel/Initializer/random_normalAddIbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/mulJbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal/mean*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes

:` 
·
:bidirectional/backward_gru/recurrent_kernel/Initializer/QrQrEbidirectional/backward_gru/recurrent_kernel/Initializer/random_normal*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*(
_output_shapes
:` :  
я
@bidirectional/backward_gru/recurrent_kernel/Initializer/DiagPartDiagPart<bidirectional/backward_gru/recurrent_kernel/Initializer/Qr:1*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes
: 
ы
<bidirectional/backward_gru/recurrent_kernel/Initializer/SignSign@bidirectional/backward_gru/recurrent_kernel/Initializer/DiagPart*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes
: 
е
;bidirectional/backward_gru/recurrent_kernel/Initializer/mulMul:bidirectional/backward_gru/recurrent_kernel/Initializer/Qr<bidirectional/backward_gru/recurrent_kernel/Initializer/Sign*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes

:` 
ш
Wbidirectional/backward_gru/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*
valueB"       *>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes
:
▐
Rbidirectional/backward_gru/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose;bidirectional/backward_gru/recurrent_kernel/Initializer/mulWbidirectional/backward_gru/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes

: `
╓
Ebidirectional/backward_gru/recurrent_kernel/Initializer/Reshape/shapeConst*
valueB"    `   *>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes
:
╬
?bidirectional/backward_gru/recurrent_kernel/Initializer/ReshapeReshapeRbidirectional/backward_gru/recurrent_kernel/Initializer/matrix_transpose/transposeEbidirectional/backward_gru/recurrent_kernel/Initializer/Reshape/shape*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes

: `
─
?bidirectional/backward_gru/recurrent_kernel/Initializer/mul_1/xConst*
valueB
 *  А?*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
п
=bidirectional/backward_gru/recurrent_kernel/Initializer/mul_1Mul?bidirectional/backward_gru/recurrent_kernel/Initializer/mul_1/x?bidirectional/backward_gru/recurrent_kernel/Initializer/Reshape*
T0*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
_output_shapes

: `
Є
+bidirectional/backward_gru/recurrent_kernelVarHandleOp*
shape
: `*<
shared_name-+bidirectional/backward_gru/recurrent_kernel*>
_class4
20loc:@bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes
: 
з
Lbidirectional/backward_gru/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+bidirectional/backward_gru/recurrent_kernel*
_output_shapes
: 
┐
2bidirectional/backward_gru/recurrent_kernel/AssignAssignVariableOp+bidirectional/backward_gru/recurrent_kernel=bidirectional/backward_gru/recurrent_kernel/Initializer/mul_1*
dtype0
л
?bidirectional/backward_gru/recurrent_kernel/Read/ReadVariableOpReadVariableOp+bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
▓
1bidirectional/backward_gru/bias/Initializer/zerosConst*
valueB`*    *2
_class(
&$loc:@bidirectional/backward_gru/bias*
dtype0*
_output_shapes
:`
╩
bidirectional/backward_gru/biasVarHandleOp*
shape:`*0
shared_name!bidirectional/backward_gru/bias*2
_class(
&$loc:@bidirectional/backward_gru/bias*
dtype0*
_output_shapes
: 
П
@bidirectional/backward_gru/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpbidirectional/backward_gru/bias*
_output_shapes
: 
Ы
&bidirectional/backward_gru/bias/AssignAssignVariableOpbidirectional/backward_gru/bias1bidirectional/backward_gru/bias/Initializer/zeros*
dtype0
П
3bidirectional/backward_gru/bias/Read/ReadVariableOpReadVariableOpbidirectional/backward_gru/bias*
dtype0*
_output_shapes
:`
v
!bidirectional/forward_gru_1/ShapeShape%embedding/embedding_lookup/Identity_1*
T0*
_output_shapes
:
y
/bidirectional/forward_gru_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
{
1bidirectional/forward_gru_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
{
1bidirectional/forward_gru_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╣
)bidirectional/forward_gru_1/strided_sliceStridedSlice!bidirectional/forward_gru_1/Shape/bidirectional/forward_gru_1/strided_slice/stack1bidirectional/forward_gru_1/strided_slice/stack_11bidirectional/forward_gru_1/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
i
'bidirectional/forward_gru_1/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: 
б
%bidirectional/forward_gru_1/zeros/mulMul)bidirectional/forward_gru_1/strided_slice'bidirectional/forward_gru_1/zeros/mul/y*
T0*
_output_shapes
: 
k
(bidirectional/forward_gru_1/zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: 
а
&bidirectional/forward_gru_1/zeros/LessLess%bidirectional/forward_gru_1/zeros/mul(bidirectional/forward_gru_1/zeros/Less/y*
T0*
_output_shapes
: 
l
*bidirectional/forward_gru_1/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: 
╡
(bidirectional/forward_gru_1/zeros/packedPack)bidirectional/forward_gru_1/strided_slice*bidirectional/forward_gru_1/zeros/packed/1*
T0*
N*
_output_shapes
:
l
'bidirectional/forward_gru_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
о
!bidirectional/forward_gru_1/zerosFill(bidirectional/forward_gru_1/zeros/packed'bidirectional/forward_gru_1/zeros/Const*
T0*'
_output_shapes
:          

*bidirectional/forward_gru_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
╜
%bidirectional/forward_gru_1/transpose	Transpose%embedding/embedding_lookup/Identity_1*bidirectional/forward_gru_1/transpose/perm*
T0*-
_output_shapes
:Ї         ╚
x
#bidirectional/forward_gru_1/Shape_1Shape%bidirectional/forward_gru_1/transpose*
T0*
_output_shapes
:
{
1bidirectional/forward_gru_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
├
+bidirectional/forward_gru_1/strided_slice_1StridedSlice#bidirectional/forward_gru_1/Shape_11bidirectional/forward_gru_1/strided_slice_1/stack3bidirectional/forward_gru_1/strided_slice_1/stack_13bidirectional/forward_gru_1/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
╙
'bidirectional/forward_gru_1/TensorArrayTensorArrayV3+bidirectional/forward_gru_1/strided_slice_1*!
tensor_array_name
input_ta_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
Й
4bidirectional/forward_gru_1/TensorArrayUnstack/ShapeShape%bidirectional/forward_gru_1/transpose*
T0*
_output_shapes
:
М
Bbidirectional/forward_gru_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
О
Dbidirectional/forward_gru_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
О
Dbidirectional/forward_gru_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ш
<bidirectional/forward_gru_1/TensorArrayUnstack/strided_sliceStridedSlice4bidirectional/forward_gru_1/TensorArrayUnstack/ShapeBbidirectional/forward_gru_1/TensorArrayUnstack/strided_slice/stackDbidirectional/forward_gru_1/TensorArrayUnstack/strided_slice/stack_1Dbidirectional/forward_gru_1/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
|
:bidirectional/forward_gru_1/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
|
:bidirectional/forward_gru_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
4bidirectional/forward_gru_1/TensorArrayUnstack/rangeRange:bidirectional/forward_gru_1/TensorArrayUnstack/range/start<bidirectional/forward_gru_1/TensorArrayUnstack/strided_slice:bidirectional/forward_gru_1/TensorArrayUnstack/range/delta*#
_output_shapes
:         
·
Vbidirectional/forward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3'bidirectional/forward_gru_1/TensorArray4bidirectional/forward_gru_1/TensorArrayUnstack/range%bidirectional/forward_gru_1/transpose)bidirectional/forward_gru_1/TensorArray:1*
T0*8
_class.
,*loc:@bidirectional/forward_gru_1/transpose*
_output_shapes
: 
{
1bidirectional/forward_gru_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╫
+bidirectional/forward_gru_1/strided_slice_2StridedSlice%bidirectional/forward_gru_1/transpose1bidirectional/forward_gru_1/strided_slice_2/stack3bidirectional/forward_gru_1/strided_slice_2/stack_13bidirectional/forward_gru_1/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*(
_output_shapes
:         ╚
Ж
+bidirectional/forward_gru_1/ones_like/ShapeShape+bidirectional/forward_gru_1/strided_slice_2*
T0*
_output_shapes
:
p
+bidirectional/forward_gru_1/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
║
%bidirectional/forward_gru_1/ones_likeFill+bidirectional/forward_gru_1/ones_like/Shape+bidirectional/forward_gru_1/ones_like/Const*
T0*(
_output_shapes
:         ╚
н
bidirectional/forward_gru_1/mulMul+bidirectional/forward_gru_1/strided_slice_2%bidirectional/forward_gru_1/ones_like*
T0*(
_output_shapes
:         ╚
п
!bidirectional/forward_gru_1/mul_1Mul+bidirectional/forward_gru_1/strided_slice_2%bidirectional/forward_gru_1/ones_like*
T0*(
_output_shapes
:         ╚
п
!bidirectional/forward_gru_1/mul_2Mul+bidirectional/forward_gru_1/strided_slice_2%bidirectional/forward_gru_1/ones_like*
T0*(
_output_shapes
:         ╚
М
*bidirectional/forward_gru_1/ReadVariableOpReadVariableOp bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
:	╚`
В
1bidirectional/forward_gru_1/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
▌
+bidirectional/forward_gru_1/strided_slice_3StridedSlice*bidirectional/forward_gru_1/ReadVariableOp1bidirectional/forward_gru_1/strided_slice_3/stack3bidirectional/forward_gru_1/strided_slice_3/stack_13bidirectional/forward_gru_1/strided_slice_3/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
м
"bidirectional/forward_gru_1/MatMulMatMulbidirectional/forward_gru_1/mul+bidirectional/forward_gru_1/strided_slice_3*
T0*'
_output_shapes
:          
О
,bidirectional/forward_gru_1/ReadVariableOp_1ReadVariableOp bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
:	╚`
В
1bidirectional/forward_gru_1/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
▀
+bidirectional/forward_gru_1/strided_slice_4StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_11bidirectional/forward_gru_1/strided_slice_4/stack3bidirectional/forward_gru_1/strided_slice_4/stack_13bidirectional/forward_gru_1/strided_slice_4/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
░
$bidirectional/forward_gru_1/MatMul_1MatMul!bidirectional/forward_gru_1/mul_1+bidirectional/forward_gru_1/strided_slice_4*
T0*'
_output_shapes
:          
О
,bidirectional/forward_gru_1/ReadVariableOp_2ReadVariableOp bidirectional/forward_gru/kernel*
dtype0*
_output_shapes
:	╚`
В
1bidirectional/forward_gru_1/strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
▀
+bidirectional/forward_gru_1/strided_slice_5StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_21bidirectional/forward_gru_1/strided_slice_5/stack3bidirectional/forward_gru_1/strided_slice_5/stack_13bidirectional/forward_gru_1/strided_slice_5/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
░
$bidirectional/forward_gru_1/MatMul_2MatMul!bidirectional/forward_gru_1/mul_2+bidirectional/forward_gru_1/strided_slice_5*
T0*'
_output_shapes
:          
З
,bidirectional/forward_gru_1/ReadVariableOp_3ReadVariableOpbidirectional/forward_gru/bias*
dtype0*
_output_shapes
:`
{
1bidirectional/forward_gru_1/strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
+bidirectional/forward_gru_1/strided_slice_6StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_31bidirectional/forward_gru_1/strided_slice_6/stack3bidirectional/forward_gru_1/strided_slice_6/stack_13bidirectional/forward_gru_1/strided_slice_6/stack_2*

begin_mask*
Index0*
T0*
_output_shapes
: 
▒
#bidirectional/forward_gru_1/BiasAddBiasAdd"bidirectional/forward_gru_1/MatMul+bidirectional/forward_gru_1/strided_slice_6*
T0*'
_output_shapes
:          
З
,bidirectional/forward_gru_1/ReadVariableOp_4ReadVariableOpbidirectional/forward_gru/bias*
dtype0*
_output_shapes
:`
{
1bidirectional/forward_gru_1/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╕
+bidirectional/forward_gru_1/strided_slice_7StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_41bidirectional/forward_gru_1/strided_slice_7/stack3bidirectional/forward_gru_1/strided_slice_7/stack_13bidirectional/forward_gru_1/strided_slice_7/stack_2*
Index0*
T0*
_output_shapes
: 
╡
%bidirectional/forward_gru_1/BiasAdd_1BiasAdd$bidirectional/forward_gru_1/MatMul_1+bidirectional/forward_gru_1/strided_slice_7*
T0*'
_output_shapes
:          
З
,bidirectional/forward_gru_1/ReadVariableOp_5ReadVariableOpbidirectional/forward_gru/bias*
dtype0*
_output_shapes
:`
{
1bidirectional/forward_gru_1/strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
}
3bidirectional/forward_gru_1/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╚
+bidirectional/forward_gru_1/strided_slice_8StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_51bidirectional/forward_gru_1/strided_slice_8/stack3bidirectional/forward_gru_1/strided_slice_8/stack_13bidirectional/forward_gru_1/strided_slice_8/stack_2*
Index0*
T0*
end_mask*
_output_shapes
: 
╡
%bidirectional/forward_gru_1/BiasAdd_2BiasAdd$bidirectional/forward_gru_1/MatMul_2+bidirectional/forward_gru_1/strided_slice_8*
T0*'
_output_shapes
:          
Ч
,bidirectional/forward_gru_1/ReadVariableOp_6ReadVariableOp*bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
В
1bidirectional/forward_gru_1/strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Д
3bidirectional/forward_gru_1/strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
▐
+bidirectional/forward_gru_1/strided_slice_9StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_61bidirectional/forward_gru_1/strided_slice_9/stack3bidirectional/forward_gru_1/strided_slice_9/stack_13bidirectional/forward_gru_1/strided_slice_9/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
░
$bidirectional/forward_gru_1/MatMul_3MatMul!bidirectional/forward_gru_1/zeros+bidirectional/forward_gru_1/strided_slice_9*
T0*'
_output_shapes
:          
Ч
,bidirectional/forward_gru_1/ReadVariableOp_7ReadVariableOp*bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
Г
2bidirectional/forward_gru_1/strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/forward_gru_1/strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
Е
4bidirectional/forward_gru_1/strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
т
,bidirectional/forward_gru_1/strided_slice_10StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_72bidirectional/forward_gru_1/strided_slice_10/stack4bidirectional/forward_gru_1/strided_slice_10/stack_14bidirectional/forward_gru_1/strided_slice_10/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
▒
$bidirectional/forward_gru_1/MatMul_4MatMul!bidirectional/forward_gru_1/zeros,bidirectional/forward_gru_1/strided_slice_10*
T0*'
_output_shapes
:          
е
bidirectional/forward_gru_1/addAddV2#bidirectional/forward_gru_1/BiasAdd$bidirectional/forward_gru_1/MatMul_3*
T0*'
_output_shapes
:          
f
!bidirectional/forward_gru_1/ConstConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
h
#bidirectional/forward_gru_1/Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ю
!bidirectional/forward_gru_1/Mul_3Mulbidirectional/forward_gru_1/add!bidirectional/forward_gru_1/Const*
T0*'
_output_shapes
:          
в
!bidirectional/forward_gru_1/Add_1Add!bidirectional/forward_gru_1/Mul_3#bidirectional/forward_gru_1/Const_1*
T0*'
_output_shapes
:          
x
3bidirectional/forward_gru_1/clip_by_value/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╞
1bidirectional/forward_gru_1/clip_by_value/MinimumMinimum!bidirectional/forward_gru_1/Add_13bidirectional/forward_gru_1/clip_by_value/Minimum/y*
T0*'
_output_shapes
:          
p
+bidirectional/forward_gru_1/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╞
)bidirectional/forward_gru_1/clip_by_valueMaximum1bidirectional/forward_gru_1/clip_by_value/Minimum+bidirectional/forward_gru_1/clip_by_value/y*
T0*'
_output_shapes
:          
й
!bidirectional/forward_gru_1/add_2AddV2%bidirectional/forward_gru_1/BiasAdd_1$bidirectional/forward_gru_1/MatMul_4*
T0*'
_output_shapes
:          
h
#bidirectional/forward_gru_1/Const_2Const*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
h
#bidirectional/forward_gru_1/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
в
!bidirectional/forward_gru_1/Mul_4Mul!bidirectional/forward_gru_1/add_2#bidirectional/forward_gru_1/Const_2*
T0*'
_output_shapes
:          
в
!bidirectional/forward_gru_1/Add_3Add!bidirectional/forward_gru_1/Mul_4#bidirectional/forward_gru_1/Const_3*
T0*'
_output_shapes
:          
z
5bidirectional/forward_gru_1/clip_by_value_1/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╩
3bidirectional/forward_gru_1/clip_by_value_1/MinimumMinimum!bidirectional/forward_gru_1/Add_35bidirectional/forward_gru_1/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:          
r
-bidirectional/forward_gru_1/clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╠
+bidirectional/forward_gru_1/clip_by_value_1Maximum3bidirectional/forward_gru_1/clip_by_value_1/Minimum-bidirectional/forward_gru_1/clip_by_value_1/y*
T0*'
_output_shapes
:          
к
!bidirectional/forward_gru_1/mul_5Mul+bidirectional/forward_gru_1/clip_by_value_1!bidirectional/forward_gru_1/zeros*
T0*'
_output_shapes
:          
Ч
,bidirectional/forward_gru_1/ReadVariableOp_8ReadVariableOp*bidirectional/forward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
Г
2bidirectional/forward_gru_1/strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
Е
4bidirectional/forward_gru_1/strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/forward_gru_1/strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
т
,bidirectional/forward_gru_1/strided_slice_11StridedSlice,bidirectional/forward_gru_1/ReadVariableOp_82bidirectional/forward_gru_1/strided_slice_11/stack4bidirectional/forward_gru_1/strided_slice_11/stack_14bidirectional/forward_gru_1/strided_slice_11/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
▒
$bidirectional/forward_gru_1/MatMul_5MatMul!bidirectional/forward_gru_1/mul_5,bidirectional/forward_gru_1/strided_slice_11*
T0*'
_output_shapes
:          
й
!bidirectional/forward_gru_1/add_4AddV2%bidirectional/forward_gru_1/BiasAdd_2$bidirectional/forward_gru_1/MatMul_5*
T0*'
_output_shapes
:          
}
 bidirectional/forward_gru_1/TanhTanh!bidirectional/forward_gru_1/add_4*
T0*'
_output_shapes
:          
и
!bidirectional/forward_gru_1/mul_6Mul)bidirectional/forward_gru_1/clip_by_value!bidirectional/forward_gru_1/zeros*
T0*'
_output_shapes
:          
f
!bidirectional/forward_gru_1/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ж
bidirectional/forward_gru_1/subSub!bidirectional/forward_gru_1/sub/x)bidirectional/forward_gru_1/clip_by_value*
T0*'
_output_shapes
:          
Э
!bidirectional/forward_gru_1/mul_7Mulbidirectional/forward_gru_1/sub bidirectional/forward_gru_1/Tanh*
T0*'
_output_shapes
:          
в
!bidirectional/forward_gru_1/add_5AddV2!bidirectional/forward_gru_1/mul_6!bidirectional/forward_gru_1/mul_7*
T0*'
_output_shapes
:          
№
)bidirectional/forward_gru_1/TensorArray_1TensorArrayV3+bidirectional/forward_gru_1/strided_slice_1*$
element_shape:          *"
tensor_array_nameoutput_ta_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
b
 bidirectional/forward_gru_1/timeConst*
value	B : *
dtype0*
_output_shapes
: 
╧
'bidirectional/forward_gru_1/while/EnterEnter bidirectional/forward_gru_1/time*
T0*
parallel_iterations *
_output_shapes
: *?

frame_name1/bidirectional/forward_gru_1/while/while_context
▄
)bidirectional/forward_gru_1/while/Enter_1Enter+bidirectional/forward_gru_1/TensorArray_1:1*
T0*
parallel_iterations *
_output_shapes
: *?

frame_name1/bidirectional/forward_gru_1/while/while_context
у
)bidirectional/forward_gru_1/while/Enter_2Enter!bidirectional/forward_gru_1/zeros*
T0*
parallel_iterations *'
_output_shapes
:          *?

frame_name1/bidirectional/forward_gru_1/while/while_context
╢
'bidirectional/forward_gru_1/while/MergeMerge'bidirectional/forward_gru_1/while/Enter/bidirectional/forward_gru_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
╝
)bidirectional/forward_gru_1/while/Merge_1Merge)bidirectional/forward_gru_1/while/Enter_11bidirectional/forward_gru_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
═
)bidirectional/forward_gru_1/while/Merge_2Merge)bidirectional/forward_gru_1/while/Enter_21bidirectional/forward_gru_1/while/NextIteration_2*
T0*
N*)
_output_shapes
:          : 
ж
&bidirectional/forward_gru_1/while/LessLess'bidirectional/forward_gru_1/while/Merge,bidirectional/forward_gru_1/while/Less/Enter*
T0*
_output_shapes
: 
Є
,bidirectional/forward_gru_1/while/Less/EnterEnter+bidirectional/forward_gru_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/bidirectional/forward_gru_1/while/while_context
v
*bidirectional/forward_gru_1/while/LoopCondLoopCond&bidirectional/forward_gru_1/while/Less*
_output_shapes
: 
ц
(bidirectional/forward_gru_1/while/SwitchSwitch'bidirectional/forward_gru_1/while/Merge*bidirectional/forward_gru_1/while/LoopCond*
T0*:
_class0
.,loc:@bidirectional/forward_gru_1/while/Merge*
_output_shapes
: : 
ь
*bidirectional/forward_gru_1/while/Switch_1Switch)bidirectional/forward_gru_1/while/Merge_1*bidirectional/forward_gru_1/while/LoopCond*
T0*<
_class2
0.loc:@bidirectional/forward_gru_1/while/Merge_1*
_output_shapes
: : 
О
*bidirectional/forward_gru_1/while/Switch_2Switch)bidirectional/forward_gru_1/while/Merge_2*bidirectional/forward_gru_1/while/LoopCond*
T0*<
_class2
0.loc:@bidirectional/forward_gru_1/while/Merge_2*:
_output_shapes(
&:          :          
Г
*bidirectional/forward_gru_1/while/IdentityIdentity*bidirectional/forward_gru_1/while/Switch:1*
T0*
_output_shapes
: 
З
,bidirectional/forward_gru_1/while/Identity_1Identity,bidirectional/forward_gru_1/while/Switch_1:1*
T0*
_output_shapes
: 
Ш
,bidirectional/forward_gru_1/while/Identity_2Identity,bidirectional/forward_gru_1/while/Switch_2:1*
T0*'
_output_shapes
:          
г
3bidirectional/forward_gru_1/while/TensorArrayReadV3TensorArrayReadV39bidirectional/forward_gru_1/while/TensorArrayReadV3/Enter*bidirectional/forward_gru_1/while/Identity;bidirectional/forward_gru_1/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         ╚
 
9bidirectional/forward_gru_1/while/TensorArrayReadV3/EnterEnter'bidirectional/forward_gru_1/TensorArray*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/bidirectional/forward_gru_1/while/while_context
м
;bidirectional/forward_gru_1/while/TensorArrayReadV3/Enter_1EnterVbidirectional/forward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/bidirectional/forward_gru_1/while/while_context
┴
%bidirectional/forward_gru_1/while/mulMul3bidirectional/forward_gru_1/while/TensorArrayReadV3+bidirectional/forward_gru_1/while/mul/Enter*
T0*(
_output_shapes
:         ╚
¤
+bidirectional/forward_gru_1/while/mul/EnterEnter%bidirectional/forward_gru_1/ones_like*
T0*
is_constant(*
parallel_iterations *(
_output_shapes
:         ╚*?

frame_name1/bidirectional/forward_gru_1/while/while_context
├
'bidirectional/forward_gru_1/while/mul_1Mul3bidirectional/forward_gru_1/while/TensorArrayReadV3+bidirectional/forward_gru_1/while/mul/Enter*
T0*(
_output_shapes
:         ╚
├
'bidirectional/forward_gru_1/while/mul_2Mul3bidirectional/forward_gru_1/while/TensorArrayReadV3+bidirectional/forward_gru_1/while/mul/Enter*
T0*(
_output_shapes
:         ╚
╒
0bidirectional/forward_gru_1/while/ReadVariableOpReadVariableOp6bidirectional/forward_gru_1/while/ReadVariableOp/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes
:	╚`
ё
6bidirectional/forward_gru_1/while/ReadVariableOp/EnterEnter bidirectional/forward_gru/kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/bidirectional/forward_gru_1/while/while_context
│
5bidirectional/forward_gru_1/while/strided_slice/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╡
7bidirectional/forward_gru_1/while/strided_slice/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╡
7bidirectional/forward_gru_1/while/strided_slice/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
є
/bidirectional/forward_gru_1/while/strided_sliceStridedSlice0bidirectional/forward_gru_1/while/ReadVariableOp5bidirectional/forward_gru_1/while/strided_slice/stack7bidirectional/forward_gru_1/while/strided_slice/stack_17bidirectional/forward_gru_1/while/strided_slice/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
╝
(bidirectional/forward_gru_1/while/MatMulMatMul%bidirectional/forward_gru_1/while/mul/bidirectional/forward_gru_1/while/strided_slice*
T0*'
_output_shapes
:          
╫
2bidirectional/forward_gru_1/while/ReadVariableOp_1ReadVariableOp6bidirectional/forward_gru_1/while/ReadVariableOp/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes
:	╚`
╡
7bidirectional/forward_gru_1/while/strided_slice_1/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_1/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_1/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
¤
1bidirectional/forward_gru_1/while/strided_slice_1StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_17bidirectional/forward_gru_1/while/strided_slice_1/stack9bidirectional/forward_gru_1/while/strided_slice_1/stack_19bidirectional/forward_gru_1/while/strided_slice_1/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
┬
*bidirectional/forward_gru_1/while/MatMul_1MatMul'bidirectional/forward_gru_1/while/mul_11bidirectional/forward_gru_1/while/strided_slice_1*
T0*'
_output_shapes
:          
╫
2bidirectional/forward_gru_1/while/ReadVariableOp_2ReadVariableOp6bidirectional/forward_gru_1/while/ReadVariableOp/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes
:	╚`
╡
7bidirectional/forward_gru_1/while/strided_slice_2/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_2/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_2/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
¤
1bidirectional/forward_gru_1/while/strided_slice_2StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_27bidirectional/forward_gru_1/while/strided_slice_2/stack9bidirectional/forward_gru_1/while/strided_slice_2/stack_19bidirectional/forward_gru_1/while/strided_slice_2/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
┬
*bidirectional/forward_gru_1/while/MatMul_2MatMul'bidirectional/forward_gru_1/while/mul_21bidirectional/forward_gru_1/while/strided_slice_2*
T0*'
_output_shapes
:          
╘
2bidirectional/forward_gru_1/while/ReadVariableOp_3ReadVariableOp8bidirectional/forward_gru_1/while/ReadVariableOp_3/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes
:`
ё
8bidirectional/forward_gru_1/while/ReadVariableOp_3/EnterEnterbidirectional/forward_gru/bias*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/bidirectional/forward_gru_1/while/while_context
о
7bidirectional/forward_gru_1/while/strided_slice_3/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
░
9bidirectional/forward_gru_1/while/strided_slice_3/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
░
9bidirectional/forward_gru_1/while/strided_slice_3/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
ш
1bidirectional/forward_gru_1/while/strided_slice_3StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_37bidirectional/forward_gru_1/while/strided_slice_3/stack9bidirectional/forward_gru_1/while/strided_slice_3/stack_19bidirectional/forward_gru_1/while/strided_slice_3/stack_2*

begin_mask*
Index0*
T0*
_output_shapes
: 
├
)bidirectional/forward_gru_1/while/BiasAddBiasAdd(bidirectional/forward_gru_1/while/MatMul1bidirectional/forward_gru_1/while/strided_slice_3*
T0*'
_output_shapes
:          
╘
2bidirectional/forward_gru_1/while/ReadVariableOp_4ReadVariableOp8bidirectional/forward_gru_1/while/ReadVariableOp_3/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes
:`
о
7bidirectional/forward_gru_1/while/strided_slice_4/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
░
9bidirectional/forward_gru_1/while/strided_slice_4/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB:@*
dtype0*
_output_shapes
:
░
9bidirectional/forward_gru_1/while/strided_slice_4/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
╓
1bidirectional/forward_gru_1/while/strided_slice_4StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_47bidirectional/forward_gru_1/while/strided_slice_4/stack9bidirectional/forward_gru_1/while/strided_slice_4/stack_19bidirectional/forward_gru_1/while/strided_slice_4/stack_2*
Index0*
T0*
_output_shapes
: 
╟
+bidirectional/forward_gru_1/while/BiasAdd_1BiasAdd*bidirectional/forward_gru_1/while/MatMul_11bidirectional/forward_gru_1/while/strided_slice_4*
T0*'
_output_shapes
:          
╘
2bidirectional/forward_gru_1/while/ReadVariableOp_5ReadVariableOp8bidirectional/forward_gru_1/while/ReadVariableOp_3/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes
:`
о
7bidirectional/forward_gru_1/while/strided_slice_5/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB:@*
dtype0*
_output_shapes
:
░
9bidirectional/forward_gru_1/while/strided_slice_5/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
░
9bidirectional/forward_gru_1/while/strided_slice_5/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
ц
1bidirectional/forward_gru_1/while/strided_slice_5StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_57bidirectional/forward_gru_1/while/strided_slice_5/stack9bidirectional/forward_gru_1/while/strided_slice_5/stack_19bidirectional/forward_gru_1/while/strided_slice_5/stack_2*
Index0*
T0*
end_mask*
_output_shapes
: 
╟
+bidirectional/forward_gru_1/while/BiasAdd_2BiasAdd*bidirectional/forward_gru_1/while/MatMul_21bidirectional/forward_gru_1/while/strided_slice_5*
T0*'
_output_shapes
:          
╪
2bidirectional/forward_gru_1/while/ReadVariableOp_6ReadVariableOp8bidirectional/forward_gru_1/while/ReadVariableOp_6/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes

: `
¤
8bidirectional/forward_gru_1/while/ReadVariableOp_6/EnterEnter*bidirectional/forward_gru/recurrent_kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/bidirectional/forward_gru_1/while/while_context
╡
7bidirectional/forward_gru_1/while/strided_slice_6/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_6/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_6/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
№
1bidirectional/forward_gru_1/while/strided_slice_6StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_67bidirectional/forward_gru_1/while/strided_slice_6/stack9bidirectional/forward_gru_1/while/strided_slice_6/stack_19bidirectional/forward_gru_1/while/strided_slice_6/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
╟
*bidirectional/forward_gru_1/while/MatMul_3MatMul,bidirectional/forward_gru_1/while/Identity_21bidirectional/forward_gru_1/while/strided_slice_6*
T0*'
_output_shapes
:          
╪
2bidirectional/forward_gru_1/while/ReadVariableOp_7ReadVariableOp8bidirectional/forward_gru_1/while/ReadVariableOp_6/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes

: `
╡
7bidirectional/forward_gru_1/while/strided_slice_7/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_7/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_7/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
№
1bidirectional/forward_gru_1/while/strided_slice_7StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_77bidirectional/forward_gru_1/while/strided_slice_7/stack9bidirectional/forward_gru_1/while/strided_slice_7/stack_19bidirectional/forward_gru_1/while/strided_slice_7/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
╟
*bidirectional/forward_gru_1/while/MatMul_4MatMul,bidirectional/forward_gru_1/while/Identity_21bidirectional/forward_gru_1/while/strided_slice_7*
T0*'
_output_shapes
:          
╖
%bidirectional/forward_gru_1/while/addAddV2)bidirectional/forward_gru_1/while/BiasAdd*bidirectional/forward_gru_1/while/MatMul_3*
T0*'
_output_shapes
:          
Щ
'bidirectional/forward_gru_1/while/ConstConst+^bidirectional/forward_gru_1/while/Identity*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ы
)bidirectional/forward_gru_1/while/Const_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
░
'bidirectional/forward_gru_1/while/Mul_3Mul%bidirectional/forward_gru_1/while/add'bidirectional/forward_gru_1/while/Const*
T0*'
_output_shapes
:          
┤
'bidirectional/forward_gru_1/while/Add_1Add'bidirectional/forward_gru_1/while/Mul_3)bidirectional/forward_gru_1/while/Const_1*
T0*'
_output_shapes
:          
л
9bidirectional/forward_gru_1/while/clip_by_value/Minimum/yConst+^bidirectional/forward_gru_1/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╪
7bidirectional/forward_gru_1/while/clip_by_value/MinimumMinimum'bidirectional/forward_gru_1/while/Add_19bidirectional/forward_gru_1/while/clip_by_value/Minimum/y*
T0*'
_output_shapes
:          
г
1bidirectional/forward_gru_1/while/clip_by_value/yConst+^bidirectional/forward_gru_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
╪
/bidirectional/forward_gru_1/while/clip_by_valueMaximum7bidirectional/forward_gru_1/while/clip_by_value/Minimum1bidirectional/forward_gru_1/while/clip_by_value/y*
T0*'
_output_shapes
:          
╗
'bidirectional/forward_gru_1/while/add_2AddV2+bidirectional/forward_gru_1/while/BiasAdd_1*bidirectional/forward_gru_1/while/MatMul_4*
T0*'
_output_shapes
:          
Ы
)bidirectional/forward_gru_1/while/Const_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ы
)bidirectional/forward_gru_1/while/Const_3Const+^bidirectional/forward_gru_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
┤
'bidirectional/forward_gru_1/while/Mul_4Mul'bidirectional/forward_gru_1/while/add_2)bidirectional/forward_gru_1/while/Const_2*
T0*'
_output_shapes
:          
┤
'bidirectional/forward_gru_1/while/Add_3Add'bidirectional/forward_gru_1/while/Mul_4)bidirectional/forward_gru_1/while/Const_3*
T0*'
_output_shapes
:          
н
;bidirectional/forward_gru_1/while/clip_by_value_1/Minimum/yConst+^bidirectional/forward_gru_1/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
▄
9bidirectional/forward_gru_1/while/clip_by_value_1/MinimumMinimum'bidirectional/forward_gru_1/while/Add_3;bidirectional/forward_gru_1/while/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:          
е
3bidirectional/forward_gru_1/while/clip_by_value_1/yConst+^bidirectional/forward_gru_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
▐
1bidirectional/forward_gru_1/while/clip_by_value_1Maximum9bidirectional/forward_gru_1/while/clip_by_value_1/Minimum3bidirectional/forward_gru_1/while/clip_by_value_1/y*
T0*'
_output_shapes
:          
┴
'bidirectional/forward_gru_1/while/mul_5Mul1bidirectional/forward_gru_1/while/clip_by_value_1,bidirectional/forward_gru_1/while/Identity_2*
T0*'
_output_shapes
:          
╪
2bidirectional/forward_gru_1/while/ReadVariableOp_8ReadVariableOp8bidirectional/forward_gru_1/while/ReadVariableOp_6/Enter+^bidirectional/forward_gru_1/while/Identity*
dtype0*
_output_shapes

: `
╡
7bidirectional/forward_gru_1/while/strided_slice_8/stackConst+^bidirectional/forward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_8/stack_1Const+^bidirectional/forward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
9bidirectional/forward_gru_1/while/strided_slice_8/stack_2Const+^bidirectional/forward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
№
1bidirectional/forward_gru_1/while/strided_slice_8StridedSlice2bidirectional/forward_gru_1/while/ReadVariableOp_87bidirectional/forward_gru_1/while/strided_slice_8/stack9bidirectional/forward_gru_1/while/strided_slice_8/stack_19bidirectional/forward_gru_1/while/strided_slice_8/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
┬
*bidirectional/forward_gru_1/while/MatMul_5MatMul'bidirectional/forward_gru_1/while/mul_51bidirectional/forward_gru_1/while/strided_slice_8*
T0*'
_output_shapes
:          
╗
'bidirectional/forward_gru_1/while/add_4AddV2+bidirectional/forward_gru_1/while/BiasAdd_2*bidirectional/forward_gru_1/while/MatMul_5*
T0*'
_output_shapes
:          
Й
&bidirectional/forward_gru_1/while/TanhTanh'bidirectional/forward_gru_1/while/add_4*
T0*'
_output_shapes
:          
┐
'bidirectional/forward_gru_1/while/mul_6Mul/bidirectional/forward_gru_1/while/clip_by_value,bidirectional/forward_gru_1/while/Identity_2*
T0*'
_output_shapes
:          
Щ
'bidirectional/forward_gru_1/while/sub/xConst+^bidirectional/forward_gru_1/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╕
%bidirectional/forward_gru_1/while/subSub'bidirectional/forward_gru_1/while/sub/x/bidirectional/forward_gru_1/while/clip_by_value*
T0*'
_output_shapes
:          
п
'bidirectional/forward_gru_1/while/mul_7Mul%bidirectional/forward_gru_1/while/sub&bidirectional/forward_gru_1/while/Tanh*
T0*'
_output_shapes
:          
┤
'bidirectional/forward_gru_1/while/add_5AddV2'bidirectional/forward_gru_1/while/mul_6'bidirectional/forward_gru_1/while/mul_7*
T0*'
_output_shapes
:          
И
Ebidirectional/forward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Kbidirectional/forward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter*bidirectional/forward_gru_1/while/Identity'bidirectional/forward_gru_1/while/add_5,bidirectional/forward_gru_1/while/Identity_1*
T0*:
_class0
.,loc:@bidirectional/forward_gru_1/while/add_5*
_output_shapes
: 
╧
Kbidirectional/forward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter)bidirectional/forward_gru_1/TensorArray_1*
T0*:
_class0
.,loc:@bidirectional/forward_gru_1/while/add_5*
parallel_iterations *
is_constant(*
_output_shapes
:*?

frame_name1/bidirectional/forward_gru_1/while/while_context
Ш
)bidirectional/forward_gru_1/while/add_6/yConst+^bidirectional/forward_gru_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
и
'bidirectional/forward_gru_1/while/add_6AddV2*bidirectional/forward_gru_1/while/Identity)bidirectional/forward_gru_1/while/add_6/y*
T0*
_output_shapes
: 
К
/bidirectional/forward_gru_1/while/NextIterationNextIteration'bidirectional/forward_gru_1/while/add_6*
T0*
_output_shapes
: 
к
1bidirectional/forward_gru_1/while/NextIteration_1NextIterationEbidirectional/forward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Э
1bidirectional/forward_gru_1/while/NextIteration_2NextIteration'bidirectional/forward_gru_1/while/add_5*
T0*'
_output_shapes
:          
y
&bidirectional/forward_gru_1/while/ExitExit(bidirectional/forward_gru_1/while/Switch*
T0*
_output_shapes
: 
}
(bidirectional/forward_gru_1/while/Exit_1Exit*bidirectional/forward_gru_1/while/Switch_1*
T0*
_output_shapes
: 
О
(bidirectional/forward_gru_1/while/Exit_2Exit*bidirectional/forward_gru_1/while/Switch_2*
T0*'
_output_shapes
:          
■
>bidirectional/forward_gru_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3)bidirectional/forward_gru_1/TensorArray_1(bidirectional/forward_gru_1/while/Exit_1*<
_class2
0.loc:@bidirectional/forward_gru_1/TensorArray_1*
_output_shapes
: 
╕
8bidirectional/forward_gru_1/TensorArrayStack/range/startConst*
value	B : *<
_class2
0.loc:@bidirectional/forward_gru_1/TensorArray_1*
dtype0*
_output_shapes
: 
╕
8bidirectional/forward_gru_1/TensorArrayStack/range/deltaConst*
value	B :*<
_class2
0.loc:@bidirectional/forward_gru_1/TensorArray_1*
dtype0*
_output_shapes
: 
╥
2bidirectional/forward_gru_1/TensorArrayStack/rangeRange8bidirectional/forward_gru_1/TensorArrayStack/range/start>bidirectional/forward_gru_1/TensorArrayStack/TensorArraySizeV38bidirectional/forward_gru_1/TensorArrayStack/range/delta*<
_class2
0.loc:@bidirectional/forward_gru_1/TensorArray_1*#
_output_shapes
:         
 
@bidirectional/forward_gru_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3)bidirectional/forward_gru_1/TensorArray_12bidirectional/forward_gru_1/TensorArrayStack/range(bidirectional/forward_gru_1/while/Exit_1*$
element_shape:          *<
_class2
0.loc:@bidirectional/forward_gru_1/TensorArray_1*
dtype0*,
_output_shapes
:Ї          
Е
2bidirectional/forward_gru_1/strided_slice_12/stackConst*
valueB:
         *
dtype0*
_output_shapes
:
~
4bidirectional/forward_gru_1/strided_slice_12/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
~
4bidirectional/forward_gru_1/strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ї
,bidirectional/forward_gru_1/strided_slice_12StridedSlice@bidirectional/forward_gru_1/TensorArrayStack/TensorArrayGatherV32bidirectional/forward_gru_1/strided_slice_12/stack4bidirectional/forward_gru_1/strided_slice_12/stack_14bidirectional/forward_gru_1/strided_slice_12/stack_2*
shrink_axis_mask*
Index0*
T0*'
_output_shapes
:          
Б
,bidirectional/forward_gru_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
█
'bidirectional/forward_gru_1/transpose_1	Transpose@bidirectional/forward_gru_1/TensorArrayStack/TensorArrayGatherV3,bidirectional/forward_gru_1/transpose_1/perm*
T0*,
_output_shapes
:         Ї 
w
"bidirectional/backward_gru_1/ShapeShape%embedding/embedding_lookup/Identity_1*
T0*
_output_shapes
:
z
0bidirectional/backward_gru_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2bidirectional/backward_gru_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2bidirectional/backward_gru_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╛
*bidirectional/backward_gru_1/strided_sliceStridedSlice"bidirectional/backward_gru_1/Shape0bidirectional/backward_gru_1/strided_slice/stack2bidirectional/backward_gru_1/strided_slice/stack_12bidirectional/backward_gru_1/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
j
(bidirectional/backward_gru_1/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: 
д
&bidirectional/backward_gru_1/zeros/mulMul*bidirectional/backward_gru_1/strided_slice(bidirectional/backward_gru_1/zeros/mul/y*
T0*
_output_shapes
: 
l
)bidirectional/backward_gru_1/zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: 
г
'bidirectional/backward_gru_1/zeros/LessLess&bidirectional/backward_gru_1/zeros/mul)bidirectional/backward_gru_1/zeros/Less/y*
T0*
_output_shapes
: 
m
+bidirectional/backward_gru_1/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: 
╕
)bidirectional/backward_gru_1/zeros/packedPack*bidirectional/backward_gru_1/strided_slice+bidirectional/backward_gru_1/zeros/packed/1*
T0*
N*
_output_shapes
:
m
(bidirectional/backward_gru_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
▒
"bidirectional/backward_gru_1/zerosFill)bidirectional/backward_gru_1/zeros/packed(bidirectional/backward_gru_1/zeros/Const*
T0*'
_output_shapes
:          
А
+bidirectional/backward_gru_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
┐
&bidirectional/backward_gru_1/transpose	Transpose%embedding/embedding_lookup/Identity_1+bidirectional/backward_gru_1/transpose/perm*
T0*-
_output_shapes
:Ї         ╚
z
$bidirectional/backward_gru_1/Shape_1Shape&bidirectional/backward_gru_1/transpose*
T0*
_output_shapes
:
|
2bidirectional/backward_gru_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╚
,bidirectional/backward_gru_1/strided_slice_1StridedSlice$bidirectional/backward_gru_1/Shape_12bidirectional/backward_gru_1/strided_slice_1/stack4bidirectional/backward_gru_1/strided_slice_1/stack_14bidirectional/backward_gru_1/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
╒
(bidirectional/backward_gru_1/TensorArrayTensorArrayV3,bidirectional/backward_gru_1/strided_slice_1*!
tensor_array_name
input_ta_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
u
+bidirectional/backward_gru_1/ReverseV2/axisConst*
valueB: *
dtype0*
_output_shapes
:
└
&bidirectional/backward_gru_1/ReverseV2	ReverseV2&bidirectional/backward_gru_1/transpose+bidirectional/backward_gru_1/ReverseV2/axis*
T0*-
_output_shapes
:Ї         ╚
Л
5bidirectional/backward_gru_1/TensorArrayUnstack/ShapeShape&bidirectional/backward_gru_1/ReverseV2*
T0*
_output_shapes
:
Н
Cbidirectional/backward_gru_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
П
Ebidirectional/backward_gru_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
П
Ebidirectional/backward_gru_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Э
=bidirectional/backward_gru_1/TensorArrayUnstack/strided_sliceStridedSlice5bidirectional/backward_gru_1/TensorArrayUnstack/ShapeCbidirectional/backward_gru_1/TensorArrayUnstack/strided_slice/stackEbidirectional/backward_gru_1/TensorArrayUnstack/strided_slice/stack_1Ebidirectional/backward_gru_1/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
}
;bidirectional/backward_gru_1/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
}
;bidirectional/backward_gru_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
5bidirectional/backward_gru_1/TensorArrayUnstack/rangeRange;bidirectional/backward_gru_1/TensorArrayUnstack/range/start=bidirectional/backward_gru_1/TensorArrayUnstack/strided_slice;bidirectional/backward_gru_1/TensorArrayUnstack/range/delta*#
_output_shapes
:         
А
Wbidirectional/backward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3(bidirectional/backward_gru_1/TensorArray5bidirectional/backward_gru_1/TensorArrayUnstack/range&bidirectional/backward_gru_1/ReverseV2*bidirectional/backward_gru_1/TensorArray:1*
T0*9
_class/
-+loc:@bidirectional/backward_gru_1/ReverseV2*
_output_shapes
: 
|
2bidirectional/backward_gru_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
▄
,bidirectional/backward_gru_1/strided_slice_2StridedSlice&bidirectional/backward_gru_1/transpose2bidirectional/backward_gru_1/strided_slice_2/stack4bidirectional/backward_gru_1/strided_slice_2/stack_14bidirectional/backward_gru_1/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*(
_output_shapes
:         ╚
И
,bidirectional/backward_gru_1/ones_like/ShapeShape,bidirectional/backward_gru_1/strided_slice_2*
T0*
_output_shapes
:
q
,bidirectional/backward_gru_1/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╜
&bidirectional/backward_gru_1/ones_likeFill,bidirectional/backward_gru_1/ones_like/Shape,bidirectional/backward_gru_1/ones_like/Const*
T0*(
_output_shapes
:         ╚
░
 bidirectional/backward_gru_1/mulMul,bidirectional/backward_gru_1/strided_slice_2&bidirectional/backward_gru_1/ones_like*
T0*(
_output_shapes
:         ╚
▓
"bidirectional/backward_gru_1/mul_1Mul,bidirectional/backward_gru_1/strided_slice_2&bidirectional/backward_gru_1/ones_like*
T0*(
_output_shapes
:         ╚
▓
"bidirectional/backward_gru_1/mul_2Mul,bidirectional/backward_gru_1/strided_slice_2&bidirectional/backward_gru_1/ones_like*
T0*(
_output_shapes
:         ╚
О
+bidirectional/backward_gru_1/ReadVariableOpReadVariableOp!bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
:	╚`
Г
2bidirectional/backward_gru_1/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
т
,bidirectional/backward_gru_1/strided_slice_3StridedSlice+bidirectional/backward_gru_1/ReadVariableOp2bidirectional/backward_gru_1/strided_slice_3/stack4bidirectional/backward_gru_1/strided_slice_3/stack_14bidirectional/backward_gru_1/strided_slice_3/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
п
#bidirectional/backward_gru_1/MatMulMatMul bidirectional/backward_gru_1/mul,bidirectional/backward_gru_1/strided_slice_3*
T0*'
_output_shapes
:          
Р
-bidirectional/backward_gru_1/ReadVariableOp_1ReadVariableOp!bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
:	╚`
Г
2bidirectional/backward_gru_1/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ф
,bidirectional/backward_gru_1/strided_slice_4StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_12bidirectional/backward_gru_1/strided_slice_4/stack4bidirectional/backward_gru_1/strided_slice_4/stack_14bidirectional/backward_gru_1/strided_slice_4/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
│
%bidirectional/backward_gru_1/MatMul_1MatMul"bidirectional/backward_gru_1/mul_1,bidirectional/backward_gru_1/strided_slice_4*
T0*'
_output_shapes
:          
Р
-bidirectional/backward_gru_1/ReadVariableOp_2ReadVariableOp!bidirectional/backward_gru/kernel*
dtype0*
_output_shapes
:	╚`
Г
2bidirectional/backward_gru_1/strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ф
,bidirectional/backward_gru_1/strided_slice_5StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_22bidirectional/backward_gru_1/strided_slice_5/stack4bidirectional/backward_gru_1/strided_slice_5/stack_14bidirectional/backward_gru_1/strided_slice_5/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
│
%bidirectional/backward_gru_1/MatMul_2MatMul"bidirectional/backward_gru_1/mul_2,bidirectional/backward_gru_1/strided_slice_5*
T0*'
_output_shapes
:          
Й
-bidirectional/backward_gru_1/ReadVariableOp_3ReadVariableOpbidirectional/backward_gru/bias*
dtype0*
_output_shapes
:`
|
2bidirectional/backward_gru_1/strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╧
,bidirectional/backward_gru_1/strided_slice_6StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_32bidirectional/backward_gru_1/strided_slice_6/stack4bidirectional/backward_gru_1/strided_slice_6/stack_14bidirectional/backward_gru_1/strided_slice_6/stack_2*

begin_mask*
Index0*
T0*
_output_shapes
: 
┤
$bidirectional/backward_gru_1/BiasAddBiasAdd#bidirectional/backward_gru_1/MatMul,bidirectional/backward_gru_1/strided_slice_6*
T0*'
_output_shapes
:          
Й
-bidirectional/backward_gru_1/ReadVariableOp_4ReadVariableOpbidirectional/backward_gru/bias*
dtype0*
_output_shapes
:`
|
2bidirectional/backward_gru_1/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_7/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╜
,bidirectional/backward_gru_1/strided_slice_7StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_42bidirectional/backward_gru_1/strided_slice_7/stack4bidirectional/backward_gru_1/strided_slice_7/stack_14bidirectional/backward_gru_1/strided_slice_7/stack_2*
Index0*
T0*
_output_shapes
: 
╕
&bidirectional/backward_gru_1/BiasAdd_1BiasAdd%bidirectional/backward_gru_1/MatMul_1,bidirectional/backward_gru_1/strided_slice_7*
T0*'
_output_shapes
:          
Й
-bidirectional/backward_gru_1/ReadVariableOp_5ReadVariableOpbidirectional/backward_gru/bias*
dtype0*
_output_shapes
:`
|
2bidirectional/backward_gru_1/strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
~
4bidirectional/backward_gru_1/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
═
,bidirectional/backward_gru_1/strided_slice_8StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_52bidirectional/backward_gru_1/strided_slice_8/stack4bidirectional/backward_gru_1/strided_slice_8/stack_14bidirectional/backward_gru_1/strided_slice_8/stack_2*
Index0*
T0*
end_mask*
_output_shapes
: 
╕
&bidirectional/backward_gru_1/BiasAdd_2BiasAdd%bidirectional/backward_gru_1/MatMul_2,bidirectional/backward_gru_1/strided_slice_8*
T0*'
_output_shapes
:          
Щ
-bidirectional/backward_gru_1/ReadVariableOp_6ReadVariableOp+bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
Г
2bidirectional/backward_gru_1/strided_slice_9/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_9/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Е
4bidirectional/backward_gru_1/strided_slice_9/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
у
,bidirectional/backward_gru_1/strided_slice_9StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_62bidirectional/backward_gru_1/strided_slice_9/stack4bidirectional/backward_gru_1/strided_slice_9/stack_14bidirectional/backward_gru_1/strided_slice_9/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
│
%bidirectional/backward_gru_1/MatMul_3MatMul"bidirectional/backward_gru_1/zeros,bidirectional/backward_gru_1/strided_slice_9*
T0*'
_output_shapes
:          
Щ
-bidirectional/backward_gru_1/ReadVariableOp_7ReadVariableOp+bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
Д
3bidirectional/backward_gru_1/strided_slice_10/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Ж
5bidirectional/backward_gru_1/strided_slice_10/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
Ж
5bidirectional/backward_gru_1/strided_slice_10/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ч
-bidirectional/backward_gru_1/strided_slice_10StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_73bidirectional/backward_gru_1/strided_slice_10/stack5bidirectional/backward_gru_1/strided_slice_10/stack_15bidirectional/backward_gru_1/strided_slice_10/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
┤
%bidirectional/backward_gru_1/MatMul_4MatMul"bidirectional/backward_gru_1/zeros-bidirectional/backward_gru_1/strided_slice_10*
T0*'
_output_shapes
:          
и
 bidirectional/backward_gru_1/addAddV2$bidirectional/backward_gru_1/BiasAdd%bidirectional/backward_gru_1/MatMul_3*
T0*'
_output_shapes
:          
g
"bidirectional/backward_gru_1/ConstConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
i
$bidirectional/backward_gru_1/Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
б
"bidirectional/backward_gru_1/Mul_3Mul bidirectional/backward_gru_1/add"bidirectional/backward_gru_1/Const*
T0*'
_output_shapes
:          
е
"bidirectional/backward_gru_1/Add_1Add"bidirectional/backward_gru_1/Mul_3$bidirectional/backward_gru_1/Const_1*
T0*'
_output_shapes
:          
y
4bidirectional/backward_gru_1/clip_by_value/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
2bidirectional/backward_gru_1/clip_by_value/MinimumMinimum"bidirectional/backward_gru_1/Add_14bidirectional/backward_gru_1/clip_by_value/Minimum/y*
T0*'
_output_shapes
:          
q
,bidirectional/backward_gru_1/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╔
*bidirectional/backward_gru_1/clip_by_valueMaximum2bidirectional/backward_gru_1/clip_by_value/Minimum,bidirectional/backward_gru_1/clip_by_value/y*
T0*'
_output_shapes
:          
м
"bidirectional/backward_gru_1/add_2AddV2&bidirectional/backward_gru_1/BiasAdd_1%bidirectional/backward_gru_1/MatMul_4*
T0*'
_output_shapes
:          
i
$bidirectional/backward_gru_1/Const_2Const*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
i
$bidirectional/backward_gru_1/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
е
"bidirectional/backward_gru_1/Mul_4Mul"bidirectional/backward_gru_1/add_2$bidirectional/backward_gru_1/Const_2*
T0*'
_output_shapes
:          
е
"bidirectional/backward_gru_1/Add_3Add"bidirectional/backward_gru_1/Mul_4$bidirectional/backward_gru_1/Const_3*
T0*'
_output_shapes
:          
{
6bidirectional/backward_gru_1/clip_by_value_1/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
═
4bidirectional/backward_gru_1/clip_by_value_1/MinimumMinimum"bidirectional/backward_gru_1/Add_36bidirectional/backward_gru_1/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:          
s
.bidirectional/backward_gru_1/clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╧
,bidirectional/backward_gru_1/clip_by_value_1Maximum4bidirectional/backward_gru_1/clip_by_value_1/Minimum.bidirectional/backward_gru_1/clip_by_value_1/y*
T0*'
_output_shapes
:          
н
"bidirectional/backward_gru_1/mul_5Mul,bidirectional/backward_gru_1/clip_by_value_1"bidirectional/backward_gru_1/zeros*
T0*'
_output_shapes
:          
Щ
-bidirectional/backward_gru_1/ReadVariableOp_8ReadVariableOp+bidirectional/backward_gru/recurrent_kernel*
dtype0*
_output_shapes

: `
Д
3bidirectional/backward_gru_1/strided_slice_11/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
Ж
5bidirectional/backward_gru_1/strided_slice_11/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
Ж
5bidirectional/backward_gru_1/strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ч
-bidirectional/backward_gru_1/strided_slice_11StridedSlice-bidirectional/backward_gru_1/ReadVariableOp_83bidirectional/backward_gru_1/strided_slice_11/stack5bidirectional/backward_gru_1/strided_slice_11/stack_15bidirectional/backward_gru_1/strided_slice_11/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
┤
%bidirectional/backward_gru_1/MatMul_5MatMul"bidirectional/backward_gru_1/mul_5-bidirectional/backward_gru_1/strided_slice_11*
T0*'
_output_shapes
:          
м
"bidirectional/backward_gru_1/add_4AddV2&bidirectional/backward_gru_1/BiasAdd_2%bidirectional/backward_gru_1/MatMul_5*
T0*'
_output_shapes
:          

!bidirectional/backward_gru_1/TanhTanh"bidirectional/backward_gru_1/add_4*
T0*'
_output_shapes
:          
л
"bidirectional/backward_gru_1/mul_6Mul*bidirectional/backward_gru_1/clip_by_value"bidirectional/backward_gru_1/zeros*
T0*'
_output_shapes
:          
g
"bidirectional/backward_gru_1/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
й
 bidirectional/backward_gru_1/subSub"bidirectional/backward_gru_1/sub/x*bidirectional/backward_gru_1/clip_by_value*
T0*'
_output_shapes
:          
а
"bidirectional/backward_gru_1/mul_7Mul bidirectional/backward_gru_1/sub!bidirectional/backward_gru_1/Tanh*
T0*'
_output_shapes
:          
е
"bidirectional/backward_gru_1/add_5AddV2"bidirectional/backward_gru_1/mul_6"bidirectional/backward_gru_1/mul_7*
T0*'
_output_shapes
:          
■
*bidirectional/backward_gru_1/TensorArray_1TensorArrayV3,bidirectional/backward_gru_1/strided_slice_1*$
element_shape:          *"
tensor_array_nameoutput_ta_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
c
!bidirectional/backward_gru_1/timeConst*
value	B : *
dtype0*
_output_shapes
: 
╥
(bidirectional/backward_gru_1/while/EnterEnter!bidirectional/backward_gru_1/time*
T0*
parallel_iterations *
_output_shapes
: *@

frame_name20bidirectional/backward_gru_1/while/while_context
▀
*bidirectional/backward_gru_1/while/Enter_1Enter,bidirectional/backward_gru_1/TensorArray_1:1*
T0*
parallel_iterations *
_output_shapes
: *@

frame_name20bidirectional/backward_gru_1/while/while_context
ц
*bidirectional/backward_gru_1/while/Enter_2Enter"bidirectional/backward_gru_1/zeros*
T0*
parallel_iterations *'
_output_shapes
:          *@

frame_name20bidirectional/backward_gru_1/while/while_context
╣
(bidirectional/backward_gru_1/while/MergeMerge(bidirectional/backward_gru_1/while/Enter0bidirectional/backward_gru_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
┐
*bidirectional/backward_gru_1/while/Merge_1Merge*bidirectional/backward_gru_1/while/Enter_12bidirectional/backward_gru_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
╨
*bidirectional/backward_gru_1/while/Merge_2Merge*bidirectional/backward_gru_1/while/Enter_22bidirectional/backward_gru_1/while/NextIteration_2*
T0*
N*)
_output_shapes
:          : 
й
'bidirectional/backward_gru_1/while/LessLess(bidirectional/backward_gru_1/while/Merge-bidirectional/backward_gru_1/while/Less/Enter*
T0*
_output_shapes
: 
ї
-bidirectional/backward_gru_1/while/Less/EnterEnter,bidirectional/backward_gru_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *@

frame_name20bidirectional/backward_gru_1/while/while_context
x
+bidirectional/backward_gru_1/while/LoopCondLoopCond'bidirectional/backward_gru_1/while/Less*
_output_shapes
: 
ъ
)bidirectional/backward_gru_1/while/SwitchSwitch(bidirectional/backward_gru_1/while/Merge+bidirectional/backward_gru_1/while/LoopCond*
T0*;
_class1
/-loc:@bidirectional/backward_gru_1/while/Merge*
_output_shapes
: : 
Ё
+bidirectional/backward_gru_1/while/Switch_1Switch*bidirectional/backward_gru_1/while/Merge_1+bidirectional/backward_gru_1/while/LoopCond*
T0*=
_class3
1/loc:@bidirectional/backward_gru_1/while/Merge_1*
_output_shapes
: : 
Т
+bidirectional/backward_gru_1/while/Switch_2Switch*bidirectional/backward_gru_1/while/Merge_2+bidirectional/backward_gru_1/while/LoopCond*
T0*=
_class3
1/loc:@bidirectional/backward_gru_1/while/Merge_2*:
_output_shapes(
&:          :          
Е
+bidirectional/backward_gru_1/while/IdentityIdentity+bidirectional/backward_gru_1/while/Switch:1*
T0*
_output_shapes
: 
Й
-bidirectional/backward_gru_1/while/Identity_1Identity-bidirectional/backward_gru_1/while/Switch_1:1*
T0*
_output_shapes
: 
Ъ
-bidirectional/backward_gru_1/while/Identity_2Identity-bidirectional/backward_gru_1/while/Switch_2:1*
T0*'
_output_shapes
:          
з
4bidirectional/backward_gru_1/while/TensorArrayReadV3TensorArrayReadV3:bidirectional/backward_gru_1/while/TensorArrayReadV3/Enter+bidirectional/backward_gru_1/while/Identity<bidirectional/backward_gru_1/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         ╚
В
:bidirectional/backward_gru_1/while/TensorArrayReadV3/EnterEnter(bidirectional/backward_gru_1/TensorArray*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20bidirectional/backward_gru_1/while/while_context
п
<bidirectional/backward_gru_1/while/TensorArrayReadV3/Enter_1EnterWbidirectional/backward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *@

frame_name20bidirectional/backward_gru_1/while/while_context
─
&bidirectional/backward_gru_1/while/mulMul4bidirectional/backward_gru_1/while/TensorArrayReadV3,bidirectional/backward_gru_1/while/mul/Enter*
T0*(
_output_shapes
:         ╚
А
,bidirectional/backward_gru_1/while/mul/EnterEnter&bidirectional/backward_gru_1/ones_like*
T0*
is_constant(*
parallel_iterations *(
_output_shapes
:         ╚*@

frame_name20bidirectional/backward_gru_1/while/while_context
╞
(bidirectional/backward_gru_1/while/mul_1Mul4bidirectional/backward_gru_1/while/TensorArrayReadV3,bidirectional/backward_gru_1/while/mul/Enter*
T0*(
_output_shapes
:         ╚
╞
(bidirectional/backward_gru_1/while/mul_2Mul4bidirectional/backward_gru_1/while/TensorArrayReadV3,bidirectional/backward_gru_1/while/mul/Enter*
T0*(
_output_shapes
:         ╚
╪
1bidirectional/backward_gru_1/while/ReadVariableOpReadVariableOp7bidirectional/backward_gru_1/while/ReadVariableOp/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes
:	╚`
Ї
7bidirectional/backward_gru_1/while/ReadVariableOp/EnterEnter!bidirectional/backward_gru/kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *@

frame_name20bidirectional/backward_gru_1/while/while_context
╡
6bidirectional/backward_gru_1/while/strided_slice/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
8bidirectional/backward_gru_1/while/strided_slice/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╖
8bidirectional/backward_gru_1/while/strided_slice/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
°
0bidirectional/backward_gru_1/while/strided_sliceStridedSlice1bidirectional/backward_gru_1/while/ReadVariableOp6bidirectional/backward_gru_1/while/strided_slice/stack8bidirectional/backward_gru_1/while/strided_slice/stack_18bidirectional/backward_gru_1/while/strided_slice/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
┐
)bidirectional/backward_gru_1/while/MatMulMatMul&bidirectional/backward_gru_1/while/mul0bidirectional/backward_gru_1/while/strided_slice*
T0*'
_output_shapes
:          
┌
3bidirectional/backward_gru_1/while/ReadVariableOp_1ReadVariableOp7bidirectional/backward_gru_1/while/ReadVariableOp/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes
:	╚`
╖
8bidirectional/backward_gru_1/while/strided_slice_1/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_1/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_1/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
В
2bidirectional/backward_gru_1/while/strided_slice_1StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_18bidirectional/backward_gru_1/while/strided_slice_1/stack:bidirectional/backward_gru_1/while/strided_slice_1/stack_1:bidirectional/backward_gru_1/while/strided_slice_1/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
┼
+bidirectional/backward_gru_1/while/MatMul_1MatMul(bidirectional/backward_gru_1/while/mul_12bidirectional/backward_gru_1/while/strided_slice_1*
T0*'
_output_shapes
:          
┌
3bidirectional/backward_gru_1/while/ReadVariableOp_2ReadVariableOp7bidirectional/backward_gru_1/while/ReadVariableOp/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes
:	╚`
╖
8bidirectional/backward_gru_1/while/strided_slice_2/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_2/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_2/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
В
2bidirectional/backward_gru_1/while/strided_slice_2StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_28bidirectional/backward_gru_1/while/strided_slice_2/stack:bidirectional/backward_gru_1/while/strided_slice_2/stack_1:bidirectional/backward_gru_1/while/strided_slice_2/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes
:	╚ 
┼
+bidirectional/backward_gru_1/while/MatMul_2MatMul(bidirectional/backward_gru_1/while/mul_22bidirectional/backward_gru_1/while/strided_slice_2*
T0*'
_output_shapes
:          
╫
3bidirectional/backward_gru_1/while/ReadVariableOp_3ReadVariableOp9bidirectional/backward_gru_1/while/ReadVariableOp_3/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes
:`
Ї
9bidirectional/backward_gru_1/while/ReadVariableOp_3/EnterEnterbidirectional/backward_gru/bias*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *@

frame_name20bidirectional/backward_gru_1/while/while_context
░
8bidirectional/backward_gru_1/while/strided_slice_3/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
▓
:bidirectional/backward_gru_1/while/strided_slice_3/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
▓
:bidirectional/backward_gru_1/while/strided_slice_3/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
э
2bidirectional/backward_gru_1/while/strided_slice_3StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_38bidirectional/backward_gru_1/while/strided_slice_3/stack:bidirectional/backward_gru_1/while/strided_slice_3/stack_1:bidirectional/backward_gru_1/while/strided_slice_3/stack_2*

begin_mask*
Index0*
T0*
_output_shapes
: 
╞
*bidirectional/backward_gru_1/while/BiasAddBiasAdd)bidirectional/backward_gru_1/while/MatMul2bidirectional/backward_gru_1/while/strided_slice_3*
T0*'
_output_shapes
:          
╫
3bidirectional/backward_gru_1/while/ReadVariableOp_4ReadVariableOp9bidirectional/backward_gru_1/while/ReadVariableOp_3/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes
:`
░
8bidirectional/backward_gru_1/while/strided_slice_4/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
▓
:bidirectional/backward_gru_1/while/strided_slice_4/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB:@*
dtype0*
_output_shapes
:
▓
:bidirectional/backward_gru_1/while/strided_slice_4/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
█
2bidirectional/backward_gru_1/while/strided_slice_4StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_48bidirectional/backward_gru_1/while/strided_slice_4/stack:bidirectional/backward_gru_1/while/strided_slice_4/stack_1:bidirectional/backward_gru_1/while/strided_slice_4/stack_2*
Index0*
T0*
_output_shapes
: 
╩
,bidirectional/backward_gru_1/while/BiasAdd_1BiasAdd+bidirectional/backward_gru_1/while/MatMul_12bidirectional/backward_gru_1/while/strided_slice_4*
T0*'
_output_shapes
:          
╫
3bidirectional/backward_gru_1/while/ReadVariableOp_5ReadVariableOp9bidirectional/backward_gru_1/while/ReadVariableOp_3/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes
:`
░
8bidirectional/backward_gru_1/while/strided_slice_5/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB:@*
dtype0*
_output_shapes
:
▓
:bidirectional/backward_gru_1/while/strided_slice_5/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
▓
:bidirectional/backward_gru_1/while/strided_slice_5/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
ы
2bidirectional/backward_gru_1/while/strided_slice_5StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_58bidirectional/backward_gru_1/while/strided_slice_5/stack:bidirectional/backward_gru_1/while/strided_slice_5/stack_1:bidirectional/backward_gru_1/while/strided_slice_5/stack_2*
Index0*
T0*
end_mask*
_output_shapes
: 
╩
,bidirectional/backward_gru_1/while/BiasAdd_2BiasAdd+bidirectional/backward_gru_1/while/MatMul_22bidirectional/backward_gru_1/while/strided_slice_5*
T0*'
_output_shapes
:          
█
3bidirectional/backward_gru_1/while/ReadVariableOp_6ReadVariableOp9bidirectional/backward_gru_1/while/ReadVariableOp_6/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes

: `
А
9bidirectional/backward_gru_1/while/ReadVariableOp_6/EnterEnter+bidirectional/backward_gru/recurrent_kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *@

frame_name20bidirectional/backward_gru_1/while/while_context
╖
8bidirectional/backward_gru_1/while/strided_slice_6/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_6/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_6/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Б
2bidirectional/backward_gru_1/while/strided_slice_6StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_68bidirectional/backward_gru_1/while/strided_slice_6/stack:bidirectional/backward_gru_1/while/strided_slice_6/stack_1:bidirectional/backward_gru_1/while/strided_slice_6/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
╩
+bidirectional/backward_gru_1/while/MatMul_3MatMul-bidirectional/backward_gru_1/while/Identity_22bidirectional/backward_gru_1/while/strided_slice_6*
T0*'
_output_shapes
:          
█
3bidirectional/backward_gru_1/while/ReadVariableOp_7ReadVariableOp9bidirectional/backward_gru_1/while/ReadVariableOp_6/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes

: `
╖
8bidirectional/backward_gru_1/while/strided_slice_7/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_7/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_7/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Б
2bidirectional/backward_gru_1/while/strided_slice_7StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_78bidirectional/backward_gru_1/while/strided_slice_7/stack:bidirectional/backward_gru_1/while/strided_slice_7/stack_1:bidirectional/backward_gru_1/while/strided_slice_7/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
╩
+bidirectional/backward_gru_1/while/MatMul_4MatMul-bidirectional/backward_gru_1/while/Identity_22bidirectional/backward_gru_1/while/strided_slice_7*
T0*'
_output_shapes
:          
║
&bidirectional/backward_gru_1/while/addAddV2*bidirectional/backward_gru_1/while/BiasAdd+bidirectional/backward_gru_1/while/MatMul_3*
T0*'
_output_shapes
:          
Ы
(bidirectional/backward_gru_1/while/ConstConst,^bidirectional/backward_gru_1/while/Identity*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Э
*bidirectional/backward_gru_1/while/Const_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
│
(bidirectional/backward_gru_1/while/Mul_3Mul&bidirectional/backward_gru_1/while/add(bidirectional/backward_gru_1/while/Const*
T0*'
_output_shapes
:          
╖
(bidirectional/backward_gru_1/while/Add_1Add(bidirectional/backward_gru_1/while/Mul_3*bidirectional/backward_gru_1/while/Const_1*
T0*'
_output_shapes
:          
н
:bidirectional/backward_gru_1/while/clip_by_value/Minimum/yConst,^bidirectional/backward_gru_1/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
█
8bidirectional/backward_gru_1/while/clip_by_value/MinimumMinimum(bidirectional/backward_gru_1/while/Add_1:bidirectional/backward_gru_1/while/clip_by_value/Minimum/y*
T0*'
_output_shapes
:          
е
2bidirectional/backward_gru_1/while/clip_by_value/yConst,^bidirectional/backward_gru_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
█
0bidirectional/backward_gru_1/while/clip_by_valueMaximum8bidirectional/backward_gru_1/while/clip_by_value/Minimum2bidirectional/backward_gru_1/while/clip_by_value/y*
T0*'
_output_shapes
:          
╛
(bidirectional/backward_gru_1/while/add_2AddV2,bidirectional/backward_gru_1/while/BiasAdd_1+bidirectional/backward_gru_1/while/MatMul_4*
T0*'
_output_shapes
:          
Э
*bidirectional/backward_gru_1/while/Const_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Э
*bidirectional/backward_gru_1/while/Const_3Const,^bidirectional/backward_gru_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
╖
(bidirectional/backward_gru_1/while/Mul_4Mul(bidirectional/backward_gru_1/while/add_2*bidirectional/backward_gru_1/while/Const_2*
T0*'
_output_shapes
:          
╖
(bidirectional/backward_gru_1/while/Add_3Add(bidirectional/backward_gru_1/while/Mul_4*bidirectional/backward_gru_1/while/Const_3*
T0*'
_output_shapes
:          
п
<bidirectional/backward_gru_1/while/clip_by_value_1/Minimum/yConst,^bidirectional/backward_gru_1/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
▀
:bidirectional/backward_gru_1/while/clip_by_value_1/MinimumMinimum(bidirectional/backward_gru_1/while/Add_3<bidirectional/backward_gru_1/while/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:          
з
4bidirectional/backward_gru_1/while/clip_by_value_1/yConst,^bidirectional/backward_gru_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
с
2bidirectional/backward_gru_1/while/clip_by_value_1Maximum:bidirectional/backward_gru_1/while/clip_by_value_1/Minimum4bidirectional/backward_gru_1/while/clip_by_value_1/y*
T0*'
_output_shapes
:          
─
(bidirectional/backward_gru_1/while/mul_5Mul2bidirectional/backward_gru_1/while/clip_by_value_1-bidirectional/backward_gru_1/while/Identity_2*
T0*'
_output_shapes
:          
█
3bidirectional/backward_gru_1/while/ReadVariableOp_8ReadVariableOp9bidirectional/backward_gru_1/while/ReadVariableOp_6/Enter,^bidirectional/backward_gru_1/while/Identity*
dtype0*
_output_shapes

: `
╖
8bidirectional/backward_gru_1/while/strided_slice_8/stackConst,^bidirectional/backward_gru_1/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_8/stack_1Const,^bidirectional/backward_gru_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
╣
:bidirectional/backward_gru_1/while/strided_slice_8/stack_2Const,^bidirectional/backward_gru_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Б
2bidirectional/backward_gru_1/while/strided_slice_8StridedSlice3bidirectional/backward_gru_1/while/ReadVariableOp_88bidirectional/backward_gru_1/while/strided_slice_8/stack:bidirectional/backward_gru_1/while/strided_slice_8/stack_1:bidirectional/backward_gru_1/while/strided_slice_8/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
┼
+bidirectional/backward_gru_1/while/MatMul_5MatMul(bidirectional/backward_gru_1/while/mul_52bidirectional/backward_gru_1/while/strided_slice_8*
T0*'
_output_shapes
:          
╛
(bidirectional/backward_gru_1/while/add_4AddV2,bidirectional/backward_gru_1/while/BiasAdd_2+bidirectional/backward_gru_1/while/MatMul_5*
T0*'
_output_shapes
:          
Л
'bidirectional/backward_gru_1/while/TanhTanh(bidirectional/backward_gru_1/while/add_4*
T0*'
_output_shapes
:          
┬
(bidirectional/backward_gru_1/while/mul_6Mul0bidirectional/backward_gru_1/while/clip_by_value-bidirectional/backward_gru_1/while/Identity_2*
T0*'
_output_shapes
:          
Ы
(bidirectional/backward_gru_1/while/sub/xConst,^bidirectional/backward_gru_1/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╗
&bidirectional/backward_gru_1/while/subSub(bidirectional/backward_gru_1/while/sub/x0bidirectional/backward_gru_1/while/clip_by_value*
T0*'
_output_shapes
:          
▓
(bidirectional/backward_gru_1/while/mul_7Mul&bidirectional/backward_gru_1/while/sub'bidirectional/backward_gru_1/while/Tanh*
T0*'
_output_shapes
:          
╖
(bidirectional/backward_gru_1/while/add_5AddV2(bidirectional/backward_gru_1/while/mul_6(bidirectional/backward_gru_1/while/mul_7*
T0*'
_output_shapes
:          
О
Fbidirectional/backward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Lbidirectional/backward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter+bidirectional/backward_gru_1/while/Identity(bidirectional/backward_gru_1/while/add_5-bidirectional/backward_gru_1/while/Identity_1*
T0*;
_class1
/-loc:@bidirectional/backward_gru_1/while/add_5*
_output_shapes
: 
╙
Lbidirectional/backward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter*bidirectional/backward_gru_1/TensorArray_1*
T0*;
_class1
/-loc:@bidirectional/backward_gru_1/while/add_5*
parallel_iterations *
is_constant(*
_output_shapes
:*@

frame_name20bidirectional/backward_gru_1/while/while_context
Ъ
*bidirectional/backward_gru_1/while/add_6/yConst,^bidirectional/backward_gru_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
л
(bidirectional/backward_gru_1/while/add_6AddV2+bidirectional/backward_gru_1/while/Identity*bidirectional/backward_gru_1/while/add_6/y*
T0*
_output_shapes
: 
М
0bidirectional/backward_gru_1/while/NextIterationNextIteration(bidirectional/backward_gru_1/while/add_6*
T0*
_output_shapes
: 
м
2bidirectional/backward_gru_1/while/NextIteration_1NextIterationFbidirectional/backward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Я
2bidirectional/backward_gru_1/while/NextIteration_2NextIteration(bidirectional/backward_gru_1/while/add_5*
T0*'
_output_shapes
:          
{
'bidirectional/backward_gru_1/while/ExitExit)bidirectional/backward_gru_1/while/Switch*
T0*
_output_shapes
: 

)bidirectional/backward_gru_1/while/Exit_1Exit+bidirectional/backward_gru_1/while/Switch_1*
T0*
_output_shapes
: 
Р
)bidirectional/backward_gru_1/while/Exit_2Exit+bidirectional/backward_gru_1/while/Switch_2*
T0*'
_output_shapes
:          
В
?bidirectional/backward_gru_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3*bidirectional/backward_gru_1/TensorArray_1)bidirectional/backward_gru_1/while/Exit_1*=
_class3
1/loc:@bidirectional/backward_gru_1/TensorArray_1*
_output_shapes
: 
║
9bidirectional/backward_gru_1/TensorArrayStack/range/startConst*
value	B : *=
_class3
1/loc:@bidirectional/backward_gru_1/TensorArray_1*
dtype0*
_output_shapes
: 
║
9bidirectional/backward_gru_1/TensorArrayStack/range/deltaConst*
value	B :*=
_class3
1/loc:@bidirectional/backward_gru_1/TensorArray_1*
dtype0*
_output_shapes
: 
╫
3bidirectional/backward_gru_1/TensorArrayStack/rangeRange9bidirectional/backward_gru_1/TensorArrayStack/range/start?bidirectional/backward_gru_1/TensorArrayStack/TensorArraySizeV39bidirectional/backward_gru_1/TensorArrayStack/range/delta*=
_class3
1/loc:@bidirectional/backward_gru_1/TensorArray_1*#
_output_shapes
:         
Д
Abidirectional/backward_gru_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3*bidirectional/backward_gru_1/TensorArray_13bidirectional/backward_gru_1/TensorArrayStack/range)bidirectional/backward_gru_1/while/Exit_1*$
element_shape:          *=
_class3
1/loc:@bidirectional/backward_gru_1/TensorArray_1*
dtype0*,
_output_shapes
:Ї          
Ж
3bidirectional/backward_gru_1/strided_slice_12/stackConst*
valueB:
         *
dtype0*
_output_shapes
:

5bidirectional/backward_gru_1/strided_slice_12/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

5bidirectional/backward_gru_1/strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
·
-bidirectional/backward_gru_1/strided_slice_12StridedSliceAbidirectional/backward_gru_1/TensorArrayStack/TensorArrayGatherV33bidirectional/backward_gru_1/strided_slice_12/stack5bidirectional/backward_gru_1/strided_slice_12/stack_15bidirectional/backward_gru_1/strided_slice_12/stack_2*
shrink_axis_mask*
Index0*
T0*'
_output_shapes
:          
В
-bidirectional/backward_gru_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
▐
(bidirectional/backward_gru_1/transpose_1	TransposeAbidirectional/backward_gru_1/TensorArrayStack/TensorArrayGatherV3-bidirectional/backward_gru_1/transpose_1/perm*
T0*,
_output_shapes
:         Ї 
f
bidirectional/ReverseV2/axisConst*
valueB:*
dtype0*
_output_shapes
:
г
bidirectional/ReverseV2	ReverseV2(bidirectional/backward_gru_1/transpose_1bidirectional/ReverseV2/axis*
T0*,
_output_shapes
:         Ї 
[
bidirectional/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
╜
bidirectional/concatConcatV2'bidirectional/forward_gru_1/transpose_1bidirectional/ReverseV2bidirectional/concat/axis*
T0*
N*,
_output_shapes
:         Ї@
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ИОЫ╛*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *ИОЫ>*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
╠
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:@
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
р
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:@
╥
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:@
Х
dense/kernelVarHandleOp*
shape
:@*
shared_namedense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:@
И
dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
Л

dense/biasVarHandleOp*
shape:*
shared_name
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
k
dense/Tensordot/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:@
^
dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
e
dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
Y
dense/Tensordot/ShapeShapebidirectional/concat*
T0*
_output_shapes
:
_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╕
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shapedense/Tensordot/freedense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
a
dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╝
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shapedense/Tensordot/axesdense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
n
dense/Tensordot/ProdProddense/Tensordot/GatherV2dense/Tensordot/Const*
T0*
_output_shapes
: 
a
dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
t
dense/Tensordot/Prod_1Proddense/Tensordot/GatherV2_1dense/Tensordot/Const_1*
T0*
_output_shapes
: 
]
dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Щ
dense/Tensordot/concatConcatV2dense/Tensordot/freedense/Tensordot/axesdense/Tensordot/concat/axis*
T0*
N*
_output_shapes
:
y
dense/Tensordot/stackPackdense/Tensordot/Proddense/Tensordot/Prod_1*
T0*
N*
_output_shapes
:
Л
dense/Tensordot/transpose	Transposebidirectional/concatdense/Tensordot/concat*
T0*,
_output_shapes
:         Ї@
П
dense/Tensordot/ReshapeReshapedense/Tensordot/transposedense/Tensordot/stack*
T0*0
_output_shapes
:                  
q
 dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
У
dense/Tensordot/transpose_1	Transposedense/Tensordot/ReadVariableOp dense/Tensordot/transpose_1/perm*
T0*
_output_shapes

:@
p
dense/Tensordot/Reshape_1/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
Л
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1dense/Tensordot/Reshape_1/shape*
T0*
_output_shapes

:@
Ж
dense/Tensordot/MatMulMatMuldense/Tensordot/Reshapedense/Tensordot/Reshape_1*
T0*'
_output_shapes
:         
a
dense/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
_
dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
д
dense/Tensordot/concat_1ConcatV2dense/Tensordot/GatherV2dense/Tensordot/Const_2dense/Tensordot/concat_1/axis*
T0*
N*
_output_shapes
:
Г
dense/TensordotReshapedense/Tensordot/MatMuldense/Tensordot/concat_1*
T0*,
_output_shapes
:         Ї
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
~
dense/BiasAddBiasAdddense/Tensordotdense/BiasAdd/ReadVariableOp*
T0*,
_output_shapes
:         Ї
X

dense/TanhTanhdense/BiasAdd*
T0*,
_output_shapes
:         Ї
G
flatten/ShapeShape
dense/Tanh*
T0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╒
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
b
flatten/Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
{
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*
N*
_output_shapes
:
p
flatten/ReshapeReshape
dense/Tanhflatten/Reshape/shape*
T0*(
_output_shapes
:         Ї
a
activation/SoftmaxSoftmaxflatten/Reshape*
T0*(
_output_shapes
:         Ї
^
repeat_vector/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
П
repeat_vector/ExpandDims
ExpandDimsactivation/Softmaxrepeat_vector/ExpandDims/dim*
T0*,
_output_shapes
:         Ї
h
repeat_vector/stackConst*!
valueB"          *
dtype0*
_output_shapes
:
А
repeat_vector/TileTilerepeat_vector/ExpandDimsrepeat_vector/stack*
T0*,
_output_shapes
:          Ї
k
permute/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Б
permute/transpose	Transposerepeat_vector/Tilepermute/transpose/perm*
T0*,
_output_shapes
:         Ї 
Y
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
а
concatenate/concatConcatV2permute/transposebidirectional/concatconcatenate/concat/axis*
T0*
N*,
_output_shapes
:         Ї`
Э
,lstm/kernel/Initializer/random_uniform/shapeConst*
valueB"`   А   *
_class
loc:@lstm/kernel*
dtype0*
_output_shapes
:
П
*lstm/kernel/Initializer/random_uniform/minConst*
valueB
 *bЧ'╛*
_class
loc:@lstm/kernel*
dtype0*
_output_shapes
: 
П
*lstm/kernel/Initializer/random_uniform/maxConst*
valueB
 *bЧ'>*
_class
loc:@lstm/kernel*
dtype0*
_output_shapes
: 
╩
4lstm/kernel/Initializer/random_uniform/RandomUniformRandomUniform,lstm/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@lstm/kernel*
dtype0*
_output_shapes
:	`А
╩
*lstm/kernel/Initializer/random_uniform/subSub*lstm/kernel/Initializer/random_uniform/max*lstm/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@lstm/kernel*
_output_shapes
: 
▌
*lstm/kernel/Initializer/random_uniform/mulMul4lstm/kernel/Initializer/random_uniform/RandomUniform*lstm/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@lstm/kernel*
_output_shapes
:	`А
╧
&lstm/kernel/Initializer/random_uniformAdd*lstm/kernel/Initializer/random_uniform/mul*lstm/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@lstm/kernel*
_output_shapes
:	`А
У
lstm/kernelVarHandleOp*
shape:	`А*
shared_namelstm/kernel*
_class
loc:@lstm/kernel*
dtype0*
_output_shapes
: 
g
,lstm/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOplstm/kernel*
_output_shapes
: 
h
lstm/kernel/AssignAssignVariableOplstm/kernel&lstm/kernel/Initializer/random_uniform*
dtype0
l
lstm/kernel/Read/ReadVariableOpReadVariableOplstm/kernel*
dtype0*
_output_shapes
:	`А
░
5lstm/recurrent_kernel/Initializer/random_normal/shapeConst*
valueB"А       *(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
:
г
4lstm/recurrent_kernel/Initializer/random_normal/meanConst*
valueB
 *    *(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
е
6lstm/recurrent_kernel/Initializer/random_normal/stddevConst*
valueB
 *  А?*(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
Ї
Dlstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal5lstm/recurrent_kernel/Initializer/random_normal/shape*
T0*(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
:	А 
М
3lstm/recurrent_kernel/Initializer/random_normal/mulMulDlstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormal6lstm/recurrent_kernel/Initializer/random_normal/stddev*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	А 
ї
/lstm/recurrent_kernel/Initializer/random_normalAdd3lstm/recurrent_kernel/Initializer/random_normal/mul4lstm/recurrent_kernel/Initializer/random_normal/mean*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	А 
╣
$lstm/recurrent_kernel/Initializer/QrQr/lstm/recurrent_kernel/Initializer/random_normal*
T0*(
_class
loc:@lstm/recurrent_kernel*)
_output_shapes
:	А :  
н
*lstm/recurrent_kernel/Initializer/DiagPartDiagPart&lstm/recurrent_kernel/Initializer/Qr:1*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
: 
й
&lstm/recurrent_kernel/Initializer/SignSign*lstm/recurrent_kernel/Initializer/DiagPart*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
: 
╬
%lstm/recurrent_kernel/Initializer/mulMul$lstm/recurrent_kernel/Initializer/Qr&lstm/recurrent_kernel/Initializer/Sign*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	А 
╝
Alstm/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*
valueB"       *(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
:
З
<lstm/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose%lstm/recurrent_kernel/Initializer/mulAlstm/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	 А
к
/lstm/recurrent_kernel/Initializer/Reshape/shapeConst*
valueB"    А   *(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
:
ў
)lstm/recurrent_kernel/Initializer/ReshapeReshape<lstm/recurrent_kernel/Initializer/matrix_transpose/transpose/lstm/recurrent_kernel/Initializer/Reshape/shape*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	 А
Ш
)lstm/recurrent_kernel/Initializer/mul_1/xConst*
valueB
 *  А?*(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
╪
'lstm/recurrent_kernel/Initializer/mul_1Mul)lstm/recurrent_kernel/Initializer/mul_1/x)lstm/recurrent_kernel/Initializer/Reshape*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	 А
▒
lstm/recurrent_kernelVarHandleOp*
shape:	 А*&
shared_namelstm/recurrent_kernel*(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
{
6lstm/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOplstm/recurrent_kernel*
_output_shapes
: 
}
lstm/recurrent_kernel/AssignAssignVariableOplstm/recurrent_kernel'lstm/recurrent_kernel/Initializer/mul_1*
dtype0
А
)lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	 А
Ж
lstm/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@lstm/bias*
dtype0*
_output_shapes
: 
Е
lstm/bias/Initializer/onesConst*
valueB *  А?*
_class
loc:@lstm/bias*
dtype0*
_output_shapes
: 
И
lstm/bias/Initializer/zeros_1Const*
valueB@*    *
_class
loc:@lstm/bias*
dtype0*
_output_shapes
:@
Б
!lstm/bias/Initializer/concat/axisConst*
value	B : *
_class
loc:@lstm/bias*
dtype0*
_output_shapes
: 
Ё
lstm/bias/Initializer/concatConcatV2lstm/bias/Initializer/zeroslstm/bias/Initializer/oneslstm/bias/Initializer/zeros_1!lstm/bias/Initializer/concat/axis*
T0*
_class
loc:@lstm/bias*
N*
_output_shapes	
:А
Й
	lstm/biasVarHandleOp*
shape:А*
shared_name	lstm/bias*
_class
loc:@lstm/bias*
dtype0*
_output_shapes
: 
c
*lstm/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp	lstm/bias*
_output_shapes
: 
Z
lstm/bias/AssignAssignVariableOp	lstm/biaslstm/bias/Initializer/concat*
dtype0
d
lstm/bias/Read/ReadVariableOpReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:А
L

lstm/ShapeShapeconcatenate/concat*
T0*
_output_shapes
:
b
lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
d
lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
d
lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╞
lstm/strided_sliceStridedSlice
lstm/Shapelstm/strided_slice/stacklstm/strided_slice/stack_1lstm/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
R
lstm/zeros/mul/yConst*
value	B : *
dtype0*
_output_shapes
: 
\
lstm/zeros/mulMullstm/strided_slicelstm/zeros/mul/y*
T0*
_output_shapes
: 
T
lstm/zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: 
[
lstm/zeros/LessLesslstm/zeros/mullstm/zeros/Less/y*
T0*
_output_shapes
: 
U
lstm/zeros/packed/1Const*
value	B : *
dtype0*
_output_shapes
: 
p
lstm/zeros/packedPacklstm/strided_slicelstm/zeros/packed/1*
T0*
N*
_output_shapes
:
U
lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i

lstm/zerosFilllstm/zeros/packedlstm/zeros/Const*
T0*'
_output_shapes
:          
T
lstm/zeros_1/mul/yConst*
value	B : *
dtype0*
_output_shapes
: 
`
lstm/zeros_1/mulMullstm/strided_slicelstm/zeros_1/mul/y*
T0*
_output_shapes
: 
V
lstm/zeros_1/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: 
a
lstm/zeros_1/LessLesslstm/zeros_1/mullstm/zeros_1/Less/y*
T0*
_output_shapes
: 
W
lstm/zeros_1/packed/1Const*
value	B : *
dtype0*
_output_shapes
: 
t
lstm/zeros_1/packedPacklstm/strided_slicelstm/zeros_1/packed/1*
T0*
N*
_output_shapes
:
W
lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
lstm/zeros_1Filllstm/zeros_1/packedlstm/zeros_1/Const*
T0*'
_output_shapes
:          
h
lstm/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
{
lstm/transpose	Transposeconcatenate/concatlstm/transpose/perm*
T0*,
_output_shapes
:Ї         `
J
lstm/Shape_1Shapelstm/transpose*
T0*
_output_shapes
:
d
lstm/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
lstm/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
lstm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╨
lstm/strided_slice_1StridedSlicelstm/Shape_1lstm/strided_slice_1/stacklstm/strided_slice_1/stack_1lstm/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
е
lstm/TensorArrayTensorArrayV3lstm/strided_slice_1*!
tensor_array_name
input_ta_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
[
lstm/TensorArrayUnstack/ShapeShapelstm/transpose*
T0*
_output_shapes
:
u
+lstm/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-lstm/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-lstm/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
е
%lstm/TensorArrayUnstack/strided_sliceStridedSlicelstm/TensorArrayUnstack/Shape+lstm/TensorArrayUnstack/strided_slice/stack-lstm/TensorArrayUnstack/strided_slice/stack_1-lstm/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
e
#lstm/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
e
#lstm/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╝
lstm/TensorArrayUnstack/rangeRange#lstm/TensorArrayUnstack/range/start%lstm/TensorArrayUnstack/strided_slice#lstm/TensorArrayUnstack/range/delta*#
_output_shapes
:         
Ё
?lstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm/TensorArraylstm/TensorArrayUnstack/rangelstm/transposelstm/TensorArray:1*
T0*!
_class
loc:@lstm/transpose*
_output_shapes
: 
d
lstm/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
lstm/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
у
lstm/strided_slice_2StridedSlicelstm/transposelstm/strided_slice_2/stacklstm/strided_slice_2/stack_1lstm/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*'
_output_shapes
:         `
X
lstm/ones_like/ShapeShapelstm/strided_slice_2*
T0*
_output_shapes
:
Y
lstm/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
t
lstm/ones_likeFilllstm/ones_like/Shapelstm/ones_like/Const*
T0*'
_output_shapes
:         `
g
lstm/mulMullstm/strided_slice_2lstm/ones_like*
T0*'
_output_shapes
:         `
i

lstm/mul_1Mullstm/strided_slice_2lstm/ones_like*
T0*'
_output_shapes
:         `
i

lstm/mul_2Mullstm/strided_slice_2lstm/ones_like*
T0*'
_output_shapes
:         `
i

lstm/mul_3Mullstm/strided_slice_2lstm/ones_like*
T0*'
_output_shapes
:         `
L

lstm/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
V
lstm/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
f
lstm/split/ReadVariableOpReadVariableOplstm/kernel*
dtype0*
_output_shapes
:	`А
Ь

lstm/splitSplitlstm/split/split_dimlstm/split/ReadVariableOp*
T0*<
_output_shapes*
(:` :` :` :` *
	num_split
]
lstm/MatMulMatMullstm/mul
lstm/split*
T0*'
_output_shapes
:          
c
lstm/MatMul_1MatMul
lstm/mul_1lstm/split:1*
T0*'
_output_shapes
:          
c
lstm/MatMul_2MatMul
lstm/mul_2lstm/split:2*
T0*'
_output_shapes
:          
c
lstm/MatMul_3MatMul
lstm/mul_3lstm/split:3*
T0*'
_output_shapes
:          
N
lstm/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
X
lstm/split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
b
lstm/split_1/ReadVariableOpReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:А
Т
lstm/split_1Splitlstm/split_1/split_dimlstm/split_1/ReadVariableOp*
T0*,
_output_shapes
: : : : *
	num_split
d
lstm/BiasAddBiasAddlstm/MatMullstm/split_1*
T0*'
_output_shapes
:          
j
lstm/BiasAdd_1BiasAddlstm/MatMul_1lstm/split_1:1*
T0*'
_output_shapes
:          
j
lstm/BiasAdd_2BiasAddlstm/MatMul_2lstm/split_1:2*
T0*'
_output_shapes
:          
j
lstm/BiasAdd_3BiasAddlstm/MatMul_3lstm/split_1:3*
T0*'
_output_shapes
:          
j
lstm/ReadVariableOpReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	 А
k
lstm/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:
m
lstm/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
m
lstm/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
щ
lstm/strided_slice_3StridedSlicelstm/ReadVariableOplstm/strided_slice_3/stacklstm/strided_slice_3/stack_1lstm/strided_slice_3/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
k
lstm/MatMul_4MatMul
lstm/zeroslstm/strided_slice_3*
T0*'
_output_shapes
:          
`
lstm/addAddV2lstm/BiasAddlstm/MatMul_4*
T0*'
_output_shapes
:          
Q
lstm/Const_2Const*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Q
lstm/Const_3Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
[

lstm/Mul_4Mullstm/addlstm/Const_2*
T0*'
_output_shapes
:          
]

lstm/Add_1Add
lstm/Mul_4lstm/Const_3*
T0*'
_output_shapes
:          
a
lstm/clip_by_value/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Б
lstm/clip_by_value/MinimumMinimum
lstm/Add_1lstm/clip_by_value/Minimum/y*
T0*'
_output_shapes
:          
Y
lstm/clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
lstm/clip_by_valueMaximumlstm/clip_by_value/Minimumlstm/clip_by_value/y*
T0*'
_output_shapes
:          
l
lstm/ReadVariableOp_1ReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	 А
k
lstm/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:
m
lstm/strided_slice_4/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
m
lstm/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ы
lstm/strided_slice_4StridedSlicelstm/ReadVariableOp_1lstm/strided_slice_4/stacklstm/strided_slice_4/stack_1lstm/strided_slice_4/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
k
lstm/MatMul_5MatMul
lstm/zeroslstm/strided_slice_4*
T0*'
_output_shapes
:          
d

lstm/add_2AddV2lstm/BiasAdd_1lstm/MatMul_5*
T0*'
_output_shapes
:          
Q
lstm/Const_4Const*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Q
lstm/Const_5Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
]

lstm/Mul_5Mul
lstm/add_2lstm/Const_4*
T0*'
_output_shapes
:          
]

lstm/Add_3Add
lstm/Mul_5lstm/Const_5*
T0*'
_output_shapes
:          
c
lstm/clip_by_value_1/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Е
lstm/clip_by_value_1/MinimumMinimum
lstm/Add_3lstm/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:          
[
lstm/clip_by_value_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
З
lstm/clip_by_value_1Maximumlstm/clip_by_value_1/Minimumlstm/clip_by_value_1/y*
T0*'
_output_shapes
:          
g

lstm/mul_6Mullstm/clip_by_value_1lstm/zeros_1*
T0*'
_output_shapes
:          
l
lstm/ReadVariableOp_2ReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	 А
k
lstm/strided_slice_5/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
m
lstm/strided_slice_5/stack_1Const*
valueB"    `   *
dtype0*
_output_shapes
:
m
lstm/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ы
lstm/strided_slice_5StridedSlicelstm/ReadVariableOp_2lstm/strided_slice_5/stacklstm/strided_slice_5/stack_1lstm/strided_slice_5/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
k
lstm/MatMul_6MatMul
lstm/zeroslstm/strided_slice_5*
T0*'
_output_shapes
:          
d

lstm/add_4AddV2lstm/BiasAdd_2lstm/MatMul_6*
T0*'
_output_shapes
:          
O
	lstm/TanhTanh
lstm/add_4*
T0*'
_output_shapes
:          
b

lstm/mul_7Mullstm/clip_by_value	lstm/Tanh*
T0*'
_output_shapes
:          
]

lstm/add_5AddV2
lstm/mul_6
lstm/mul_7*
T0*'
_output_shapes
:          
l
lstm/ReadVariableOp_3ReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	 А
k
lstm/strided_slice_6/stackConst*
valueB"    `   *
dtype0*
_output_shapes
:
m
lstm/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
m
lstm/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ы
lstm/strided_slice_6StridedSlicelstm/ReadVariableOp_3lstm/strided_slice_6/stacklstm/strided_slice_6/stack_1lstm/strided_slice_6/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
k
lstm/MatMul_7MatMul
lstm/zeroslstm/strided_slice_6*
T0*'
_output_shapes
:          
d

lstm/add_6AddV2lstm/BiasAdd_3lstm/MatMul_7*
T0*'
_output_shapes
:          
Q
lstm/Const_6Const*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Q
lstm/Const_7Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
]

lstm/Mul_8Mul
lstm/add_6lstm/Const_6*
T0*'
_output_shapes
:          
]

lstm/Add_7Add
lstm/Mul_8lstm/Const_7*
T0*'
_output_shapes
:          
c
lstm/clip_by_value_2/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Е
lstm/clip_by_value_2/MinimumMinimum
lstm/Add_7lstm/clip_by_value_2/Minimum/y*
T0*'
_output_shapes
:          
[
lstm/clip_by_value_2/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
З
lstm/clip_by_value_2Maximumlstm/clip_by_value_2/Minimumlstm/clip_by_value_2/y*
T0*'
_output_shapes
:          
Q
lstm/Tanh_1Tanh
lstm/add_5*
T0*'
_output_shapes
:          
f

lstm/mul_9Mullstm/clip_by_value_2lstm/Tanh_1*
T0*'
_output_shapes
:          
╬
lstm/TensorArray_1TensorArrayV3lstm/strided_slice_1*$
element_shape:          *"
tensor_array_nameoutput_ta_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
K
	lstm/timeConst*
value	B : *
dtype0*
_output_shapes
: 
К
lstm/while/EnterEnter	lstm/time*
T0*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
Ч
lstm/while/Enter_1Enterlstm/TensorArray_1:1*
T0*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
Ю
lstm/while/Enter_2Enter
lstm/zeros*
T0*
parallel_iterations *'
_output_shapes
:          *(

frame_namelstm/while/while_context
а
lstm/while/Enter_3Enterlstm/zeros_1*
T0*
parallel_iterations *'
_output_shapes
:          *(

frame_namelstm/while/while_context
q
lstm/while/MergeMergelstm/while/Enterlstm/while/NextIteration*
T0*
N*
_output_shapes
: : 
w
lstm/while/Merge_1Mergelstm/while/Enter_1lstm/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
И
lstm/while/Merge_2Mergelstm/while/Enter_2lstm/while/NextIteration_2*
T0*
N*)
_output_shapes
:          : 
И
lstm/while/Merge_3Mergelstm/while/Enter_3lstm/while/NextIteration_3*
T0*
N*)
_output_shapes
:          : 
a
lstm/while/LessLesslstm/while/Mergelstm/while/Less/Enter*
T0*
_output_shapes
: 
н
lstm/while/Less/EnterEnterlstm/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
H
lstm/while/LoopCondLoopCondlstm/while/Less*
_output_shapes
: 
К
lstm/while/SwitchSwitchlstm/while/Mergelstm/while/LoopCond*
T0*#
_class
loc:@lstm/while/Merge*
_output_shapes
: : 
Р
lstm/while/Switch_1Switchlstm/while/Merge_1lstm/while/LoopCond*
T0*%
_class
loc:@lstm/while/Merge_1*
_output_shapes
: : 
▓
lstm/while/Switch_2Switchlstm/while/Merge_2lstm/while/LoopCond*
T0*%
_class
loc:@lstm/while/Merge_2*:
_output_shapes(
&:          :          
▓
lstm/while/Switch_3Switchlstm/while/Merge_3lstm/while/LoopCond*
T0*%
_class
loc:@lstm/while/Merge_3*:
_output_shapes(
&:          :          
U
lstm/while/IdentityIdentitylstm/while/Switch:1*
T0*
_output_shapes
: 
Y
lstm/while/Identity_1Identitylstm/while/Switch_1:1*
T0*
_output_shapes
: 
j
lstm/while/Identity_2Identitylstm/while/Switch_2:1*
T0*'
_output_shapes
:          
j
lstm/while/Identity_3Identitylstm/while/Switch_3:1*
T0*'
_output_shapes
:          
╞
lstm/while/TensorArrayReadV3TensorArrayReadV3"lstm/while/TensorArrayReadV3/Enterlstm/while/Identity$lstm/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         `
║
"lstm/while/TensorArrayReadV3/EnterEnterlstm/TensorArray*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*(

frame_namelstm/while/while_context
ч
$lstm/while/TensorArrayReadV3/Enter_1Enter?lstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
{
lstm/while/mulMullstm/while/TensorArrayReadV3lstm/while/mul/Enter*
T0*'
_output_shapes
:         `
╖
lstm/while/mul/EnterEnterlstm/ones_like*
T0*
is_constant(*
parallel_iterations *'
_output_shapes
:         `*(

frame_namelstm/while/while_context
}
lstm/while/mul_1Mullstm/while/TensorArrayReadV3lstm/while/mul/Enter*
T0*'
_output_shapes
:         `
}
lstm/while/mul_2Mullstm/while/TensorArrayReadV3lstm/while/mul/Enter*
T0*'
_output_shapes
:         `
}
lstm/while/mul_3Mullstm/while/TensorArrayReadV3lstm/while/mul/Enter*
T0*'
_output_shapes
:         `
h
lstm/while/ConstConst^lstm/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
r
lstm/while/split/split_dimConst^lstm/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ь
lstm/while/split/ReadVariableOpReadVariableOp%lstm/while/split/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	`А
┤
%lstm/while/split/ReadVariableOp/EnterEnterlstm/kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
о
lstm/while/splitSplitlstm/while/split/split_dimlstm/while/split/ReadVariableOp*
T0*<
_output_shapes*
(:` :` :` :` *
	num_split
o
lstm/while/MatMulMatMullstm/while/mullstm/while/split*
T0*'
_output_shapes
:          
u
lstm/while/MatMul_1MatMullstm/while/mul_1lstm/while/split:1*
T0*'
_output_shapes
:          
u
lstm/while/MatMul_2MatMullstm/while/mul_2lstm/while/split:2*
T0*'
_output_shapes
:          
u
lstm/while/MatMul_3MatMullstm/while/mul_3lstm/while/split:3*
T0*'
_output_shapes
:          
j
lstm/while/Const_1Const^lstm/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
t
lstm/while/split_1/split_dimConst^lstm/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Ь
!lstm/while/split_1/ReadVariableOpReadVariableOp'lstm/while/split_1/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes	
:А
┤
'lstm/while/split_1/ReadVariableOp/EnterEnter	lstm/bias*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
д
lstm/while/split_1Splitlstm/while/split_1/split_dim!lstm/while/split_1/ReadVariableOp*
T0*,
_output_shapes
: : : : *
	num_split
v
lstm/while/BiasAddBiasAddlstm/while/MatMullstm/while/split_1*
T0*'
_output_shapes
:          
|
lstm/while/BiasAdd_1BiasAddlstm/while/MatMul_1lstm/while/split_1:1*
T0*'
_output_shapes
:          
|
lstm/while/BiasAdd_2BiasAddlstm/while/MatMul_2lstm/while/split_1:2*
T0*'
_output_shapes
:          
|
lstm/while/BiasAdd_3BiasAddlstm/while/MatMul_3lstm/while/split_1:3*
T0*'
_output_shapes
:          
Р
lstm/while/ReadVariableOpReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	 А
╕
lstm/while/ReadVariableOp/EnterEnterlstm/recurrent_kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
Е
lstm/while/strided_slice/stackConst^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
З
 lstm/while/strided_slice/stack_1Const^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
З
 lstm/while/strided_slice/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
 
lstm/while/strided_sliceStridedSlicelstm/while/ReadVariableOplstm/while/strided_slice/stack lstm/while/strided_slice/stack_1 lstm/while/strided_slice/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
А
lstm/while/MatMul_4MatMullstm/while/Identity_2lstm/while/strided_slice*
T0*'
_output_shapes
:          
r
lstm/while/addAddV2lstm/while/BiasAddlstm/while/MatMul_4*
T0*'
_output_shapes
:          
m
lstm/while/Const_2Const^lstm/while/Identity*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
m
lstm/while/Const_3Const^lstm/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
m
lstm/while/Mul_4Mullstm/while/addlstm/while/Const_2*
T0*'
_output_shapes
:          
o
lstm/while/Add_1Addlstm/while/Mul_4lstm/while/Const_3*
T0*'
_output_shapes
:          
}
"lstm/while/clip_by_value/Minimum/yConst^lstm/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
У
 lstm/while/clip_by_value/MinimumMinimumlstm/while/Add_1"lstm/while/clip_by_value/Minimum/y*
T0*'
_output_shapes
:          
u
lstm/while/clip_by_value/yConst^lstm/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
У
lstm/while/clip_by_valueMaximum lstm/while/clip_by_value/Minimumlstm/while/clip_by_value/y*
T0*'
_output_shapes
:          
Т
lstm/while/ReadVariableOp_1ReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	 А
З
 lstm/while/strided_slice_1/stackConst^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
Й
"lstm/while/strided_slice_1/stack_1Const^lstm/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
Й
"lstm/while/strided_slice_1/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Й
lstm/while/strided_slice_1StridedSlicelstm/while/ReadVariableOp_1 lstm/while/strided_slice_1/stack"lstm/while/strided_slice_1/stack_1"lstm/while/strided_slice_1/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
В
lstm/while/MatMul_5MatMullstm/while/Identity_2lstm/while/strided_slice_1*
T0*'
_output_shapes
:          
v
lstm/while/add_2AddV2lstm/while/BiasAdd_1lstm/while/MatMul_5*
T0*'
_output_shapes
:          
m
lstm/while/Const_4Const^lstm/while/Identity*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
m
lstm/while/Const_5Const^lstm/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
o
lstm/while/Mul_5Mullstm/while/add_2lstm/while/Const_4*
T0*'
_output_shapes
:          
o
lstm/while/Add_3Addlstm/while/Mul_5lstm/while/Const_5*
T0*'
_output_shapes
:          

$lstm/while/clip_by_value_1/Minimum/yConst^lstm/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ч
"lstm/while/clip_by_value_1/MinimumMinimumlstm/while/Add_3$lstm/while/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:          
w
lstm/while/clip_by_value_1/yConst^lstm/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
Щ
lstm/while/clip_by_value_1Maximum"lstm/while/clip_by_value_1/Minimumlstm/while/clip_by_value_1/y*
T0*'
_output_shapes
:          
|
lstm/while/mul_6Mullstm/while/clip_by_value_1lstm/while/Identity_3*
T0*'
_output_shapes
:          
Т
lstm/while/ReadVariableOp_2ReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	 А
З
 lstm/while/strided_slice_2/stackConst^lstm/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:
Й
"lstm/while/strided_slice_2/stack_1Const^lstm/while/Identity*
valueB"    `   *
dtype0*
_output_shapes
:
Й
"lstm/while/strided_slice_2/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Й
lstm/while/strided_slice_2StridedSlicelstm/while/ReadVariableOp_2 lstm/while/strided_slice_2/stack"lstm/while/strided_slice_2/stack_1"lstm/while/strided_slice_2/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
В
lstm/while/MatMul_6MatMullstm/while/Identity_2lstm/while/strided_slice_2*
T0*'
_output_shapes
:          
v
lstm/while/add_4AddV2lstm/while/BiasAdd_2lstm/while/MatMul_6*
T0*'
_output_shapes
:          
[
lstm/while/TanhTanhlstm/while/add_4*
T0*'
_output_shapes
:          
t
lstm/while/mul_7Mullstm/while/clip_by_valuelstm/while/Tanh*
T0*'
_output_shapes
:          
o
lstm/while/add_5AddV2lstm/while/mul_6lstm/while/mul_7*
T0*'
_output_shapes
:          
Т
lstm/while/ReadVariableOp_3ReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	 А
З
 lstm/while/strided_slice_3/stackConst^lstm/while/Identity*
valueB"    `   *
dtype0*
_output_shapes
:
Й
"lstm/while/strided_slice_3/stack_1Const^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:
Й
"lstm/while/strided_slice_3/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Й
lstm/while/strided_slice_3StridedSlicelstm/while/ReadVariableOp_3 lstm/while/strided_slice_3/stack"lstm/while/strided_slice_3/stack_1"lstm/while/strided_slice_3/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:  
В
lstm/while/MatMul_7MatMullstm/while/Identity_2lstm/while/strided_slice_3*
T0*'
_output_shapes
:          
v
lstm/while/add_6AddV2lstm/while/BiasAdd_3lstm/while/MatMul_7*
T0*'
_output_shapes
:          
m
lstm/while/Const_6Const^lstm/while/Identity*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
m
lstm/while/Const_7Const^lstm/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
o
lstm/while/Mul_8Mullstm/while/add_6lstm/while/Const_6*
T0*'
_output_shapes
:          
o
lstm/while/Add_7Addlstm/while/Mul_8lstm/while/Const_7*
T0*'
_output_shapes
:          

$lstm/while/clip_by_value_2/Minimum/yConst^lstm/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ч
"lstm/while/clip_by_value_2/MinimumMinimumlstm/while/Add_7$lstm/while/clip_by_value_2/Minimum/y*
T0*'
_output_shapes
:          
w
lstm/while/clip_by_value_2/yConst^lstm/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
Щ
lstm/while/clip_by_value_2Maximum"lstm/while/clip_by_value_2/Minimumlstm/while/clip_by_value_2/y*
T0*'
_output_shapes
:          
]
lstm/while/Tanh_1Tanhlstm/while/add_5*
T0*'
_output_shapes
:          
x
lstm/while/mul_9Mullstm/while/clip_by_value_2lstm/while/Tanh_1*
T0*'
_output_shapes
:          
■
.lstm/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV34lstm/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm/while/Identitylstm/while/mul_9lstm/while/Identity_1*
T0*#
_class
loc:@lstm/while/mul_9*
_output_shapes
: 
є
4lstm/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm/TensorArray_1*
T0*#
_class
loc:@lstm/while/mul_9*
parallel_iterations *
is_constant(*
_output_shapes
:*(

frame_namelstm/while/while_context
j
lstm/while/add_8/yConst^lstm/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
lstm/while/add_8AddV2lstm/while/Identitylstm/while/add_8/y*
T0*
_output_shapes
: 
\
lstm/while/NextIterationNextIterationlstm/while/add_8*
T0*
_output_shapes
: 
|
lstm/while/NextIteration_1NextIteration.lstm/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
o
lstm/while/NextIteration_2NextIterationlstm/while/mul_9*
T0*'
_output_shapes
:          
o
lstm/while/NextIteration_3NextIterationlstm/while/add_5*
T0*'
_output_shapes
:          
K
lstm/while/ExitExitlstm/while/Switch*
T0*
_output_shapes
: 
O
lstm/while/Exit_1Exitlstm/while/Switch_1*
T0*
_output_shapes
: 
`
lstm/while/Exit_2Exitlstm/while/Switch_2*
T0*'
_output_shapes
:          
`
lstm/while/Exit_3Exitlstm/while/Switch_3*
T0*'
_output_shapes
:          
в
'lstm/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm/TensorArray_1lstm/while/Exit_1*%
_class
loc:@lstm/TensorArray_1*
_output_shapes
: 
К
!lstm/TensorArrayStack/range/startConst*
value	B : *%
_class
loc:@lstm/TensorArray_1*
dtype0*
_output_shapes
: 
К
!lstm/TensorArrayStack/range/deltaConst*
value	B :*%
_class
loc:@lstm/TensorArray_1*
dtype0*
_output_shapes
: 
▀
lstm/TensorArrayStack/rangeRange!lstm/TensorArrayStack/range/start'lstm/TensorArrayStack/TensorArraySizeV3!lstm/TensorArrayStack/range/delta*%
_class
loc:@lstm/TensorArray_1*#
_output_shapes
:         
М
)lstm/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm/TensorArray_1lstm/TensorArrayStack/rangelstm/while/Exit_1*$
element_shape:          *%
_class
loc:@lstm/TensorArray_1*
dtype0*,
_output_shapes
:Ї          
m
lstm/strided_slice_7/stackConst*
valueB:
         *
dtype0*
_output_shapes
:
f
lstm/strided_slice_7/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
f
lstm/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
■
lstm/strided_slice_7StridedSlice)lstm/TensorArrayStack/TensorArrayGatherV3lstm/strided_slice_7/stacklstm/strided_slice_7/stack_1lstm/strided_slice_7/stack_2*
shrink_axis_mask*
Index0*
T0*'
_output_shapes
:          
j
lstm/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Ц
lstm/transpose_1	Transpose)lstm/TensorArrayStack/TensorArrayGatherV3lstm/transpose_1/perm*
T0*,
_output_shapes
:         Ї 
б
.output/kernel/Initializer/random_uniform/shapeConst*
valueB"       * 
_class
loc:@output/kernel*
dtype0*
_output_shapes
:
У
,output/kernel/Initializer/random_uniform/minConst*
valueB
 *A╫╛* 
_class
loc:@output/kernel*
dtype0*
_output_shapes
: 
У
,output/kernel/Initializer/random_uniform/maxConst*
valueB
 *A╫>* 
_class
loc:@output/kernel*
dtype0*
_output_shapes
: 
╧
6output/kernel/Initializer/random_uniform/RandomUniformRandomUniform.output/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@output/kernel*
dtype0*
_output_shapes

: 
╥
,output/kernel/Initializer/random_uniform/subSub,output/kernel/Initializer/random_uniform/max,output/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@output/kernel*
_output_shapes
: 
ф
,output/kernel/Initializer/random_uniform/mulMul6output/kernel/Initializer/random_uniform/RandomUniform,output/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@output/kernel*
_output_shapes

: 
╓
(output/kernel/Initializer/random_uniformAdd,output/kernel/Initializer/random_uniform/mul,output/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@output/kernel*
_output_shapes

: 
Ш
output/kernelVarHandleOp*
shape
: *
shared_nameoutput/kernel* 
_class
loc:@output/kernel*
dtype0*
_output_shapes
: 
k
.output/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpoutput/kernel*
_output_shapes
: 
n
output/kernel/AssignAssignVariableOpoutput/kernel(output/kernel/Initializer/random_uniform*
dtype0
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
dtype0*
_output_shapes

: 
К
output/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@output/bias*
dtype0*
_output_shapes
:
О
output/biasVarHandleOp*
shape:*
shared_nameoutput/bias*
_class
loc:@output/bias*
dtype0*
_output_shapes
: 
g
,output/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpoutput/bias*
_output_shapes
: 
_
output/bias/AssignAssignVariableOpoutput/biasoutput/bias/Initializer/zeros*
dtype0
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
dtype0*
_output_shapes
:
j
output/MatMul/ReadVariableOpReadVariableOpoutput/kernel*
dtype0*
_output_shapes

: 
}
output/MatMulMatMullstm/strided_slice_7output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
e
output/BiasAdd/ReadVariableOpReadVariableOpoutput/bias*
dtype0*
_output_shapes
:
y
output/BiasAddBiasAddoutput/MatMuloutput/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:         
[
output/SoftmaxSoftmaxoutput/BiasAdd*
T0*'
_output_shapes
:         

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_09d78abfb09f490eab29754c20950021/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
╪
save/SaveV2/tensor_namesConst"/device:CPU:0*№
valueЄBяBbidirectional/backward_gru/biasB!bidirectional/backward_gru/kernelB+bidirectional/backward_gru/recurrent_kernelBbidirectional/forward_gru/biasB bidirectional/forward_gru/kernelB*bidirectional/forward_gru/recurrent_kernelB
dense/biasBdense/kernelBembedding/embeddingsBglobal_stepB	lstm/biasBlstm/kernelBlstm/recurrent_kernelBoutput/biasBoutput/kernel*
dtype0*
_output_shapes
:
Р
save/SaveV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Р
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices3bidirectional/backward_gru/bias/Read/ReadVariableOp5bidirectional/backward_gru/kernel/Read/ReadVariableOp?bidirectional/backward_gru/recurrent_kernel/Read/ReadVariableOp2bidirectional/forward_gru/bias/Read/ReadVariableOp4bidirectional/forward_gru/kernel/Read/ReadVariableOp>bidirectional/forward_gru/recurrent_kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOpglobal_steplstm/bias/Read/ReadVariableOplstm/kernel/Read/ReadVariableOp)lstm/recurrent_kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOp"/device:CPU:0*
dtypes
2	
а
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
а
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
█
save/RestoreV2/tensor_namesConst"/device:CPU:0*№
valueЄBяBbidirectional/backward_gru/biasB!bidirectional/backward_gru/kernelB+bidirectional/backward_gru/recurrent_kernelBbidirectional/forward_gru/biasB bidirectional/forward_gru/kernelB*bidirectional/forward_gru/recurrent_kernelB
dense/biasBdense/kernelBembedding/embeddingsBglobal_stepB	lstm/biasBlstm/kernelBlstm/recurrent_kernelBoutput/biasBoutput/kernel*
dtype0*
_output_shapes
:
У
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
х
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*P
_output_shapes>
<:::::::::::::::
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
h
save/AssignVariableOpAssignVariableOpbidirectional/backward_gru/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
l
save/AssignVariableOp_1AssignVariableOp!bidirectional/backward_gru/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
v
save/AssignVariableOp_2AssignVariableOp+bidirectional/backward_gru/recurrent_kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
i
save/AssignVariableOp_3AssignVariableOpbidirectional/forward_gru/biassave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
k
save/AssignVariableOp_4AssignVariableOp bidirectional/forward_gru/kernelsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
u
save/AssignVariableOp_5AssignVariableOp*bidirectional/forward_gru/recurrent_kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
U
save/AssignVariableOp_6AssignVariableOp
dense/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
W
save/AssignVariableOp_7AssignVariableOpdense/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
_
save/AssignVariableOp_8AssignVariableOpembedding/embeddingssave/Identity_9*
dtype0
u
save/AssignAssignglobal_stepsave/RestoreV2:9*
T0	*
_class
loc:@global_step*
_output_shapes
: 
R
save/Identity_10Identitysave/RestoreV2:10*
T0*
_output_shapes
:
U
save/AssignVariableOp_9AssignVariableOp	lstm/biassave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:11*
T0*
_output_shapes
:
X
save/AssignVariableOp_10AssignVariableOplstm/kernelsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:12*
T0*
_output_shapes
:
b
save/AssignVariableOp_11AssignVariableOplstm/recurrent_kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:13*
T0*
_output_shapes
:
X
save/AssignVariableOp_12AssignVariableOpoutput/biassave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:14*
T0*
_output_shapes
:
Z
save/AssignVariableOp_13AssignVariableOpoutput/kernelsave/Identity_14*
dtype0
Ц
save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard"Ж<
save/Const:0save/Identity:0save/restore_all (5 @F8"р
trainable_variables╚┼
Ш
embedding/embeddings:0embedding/embeddings/Assign*embedding/embeddings/Read/ReadVariableOp:0(21embedding/embeddings/Initializer/random_uniform:08
╚
"bidirectional/forward_gru/kernel:0'bidirectional/forward_gru/kernel/Assign6bidirectional/forward_gru/kernel/Read/ReadVariableOp:0(2=bidirectional/forward_gru/kernel/Initializer/random_uniform:08
ч
,bidirectional/forward_gru/recurrent_kernel:01bidirectional/forward_gru/recurrent_kernel/Assign@bidirectional/forward_gru/recurrent_kernel/Read/ReadVariableOp:0(2>bidirectional/forward_gru/recurrent_kernel/Initializer/mul_1:08
╖
 bidirectional/forward_gru/bias:0%bidirectional/forward_gru/bias/Assign4bidirectional/forward_gru/bias/Read/ReadVariableOp:0(22bidirectional/forward_gru/bias/Initializer/zeros:08
╠
#bidirectional/backward_gru/kernel:0(bidirectional/backward_gru/kernel/Assign7bidirectional/backward_gru/kernel/Read/ReadVariableOp:0(2>bidirectional/backward_gru/kernel/Initializer/random_uniform:08
ы
-bidirectional/backward_gru/recurrent_kernel:02bidirectional/backward_gru/recurrent_kernel/AssignAbidirectional/backward_gru/recurrent_kernel/Read/ReadVariableOp:0(2?bidirectional/backward_gru/recurrent_kernel/Initializer/mul_1:08
╗
!bidirectional/backward_gru/bias:0&bidirectional/backward_gru/bias/Assign5bidirectional/backward_gru/bias/Read/ReadVariableOp:0(23bidirectional/backward_gru/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
t
lstm/kernel:0lstm/kernel/Assign!lstm/kernel/Read/ReadVariableOp:0(2(lstm/kernel/Initializer/random_uniform:08
У
lstm/recurrent_kernel:0lstm/recurrent_kernel/Assign+lstm/recurrent_kernel/Read/ReadVariableOp:0(2)lstm/recurrent_kernel/Initializer/mul_1:08
d
lstm/bias:0lstm/bias/Assignlstm/bias/Read/ReadVariableOp:0(2lstm/bias/Initializer/concat:08
|
output/kernel:0output/kernel/Assign#output/kernel/Read/ReadVariableOp:0(2*output/kernel/Initializer/random_uniform:08
k
output/bias:0output/bias/Assign!output/bias/Read/ReadVariableOp:0(2output/bias/Initializer/zeros:08"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"▓
	variablesдб
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
Ш
embedding/embeddings:0embedding/embeddings/Assign*embedding/embeddings/Read/ReadVariableOp:0(21embedding/embeddings/Initializer/random_uniform:08
╚
"bidirectional/forward_gru/kernel:0'bidirectional/forward_gru/kernel/Assign6bidirectional/forward_gru/kernel/Read/ReadVariableOp:0(2=bidirectional/forward_gru/kernel/Initializer/random_uniform:08
ч
,bidirectional/forward_gru/recurrent_kernel:01bidirectional/forward_gru/recurrent_kernel/Assign@bidirectional/forward_gru/recurrent_kernel/Read/ReadVariableOp:0(2>bidirectional/forward_gru/recurrent_kernel/Initializer/mul_1:08
╖
 bidirectional/forward_gru/bias:0%bidirectional/forward_gru/bias/Assign4bidirectional/forward_gru/bias/Read/ReadVariableOp:0(22bidirectional/forward_gru/bias/Initializer/zeros:08
╠
#bidirectional/backward_gru/kernel:0(bidirectional/backward_gru/kernel/Assign7bidirectional/backward_gru/kernel/Read/ReadVariableOp:0(2>bidirectional/backward_gru/kernel/Initializer/random_uniform:08
ы
-bidirectional/backward_gru/recurrent_kernel:02bidirectional/backward_gru/recurrent_kernel/AssignAbidirectional/backward_gru/recurrent_kernel/Read/ReadVariableOp:0(2?bidirectional/backward_gru/recurrent_kernel/Initializer/mul_1:08
╗
!bidirectional/backward_gru/bias:0&bidirectional/backward_gru/bias/Assign5bidirectional/backward_gru/bias/Read/ReadVariableOp:0(23bidirectional/backward_gru/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
t
lstm/kernel:0lstm/kernel/Assign!lstm/kernel/Read/ReadVariableOp:0(2(lstm/kernel/Initializer/random_uniform:08
У
lstm/recurrent_kernel:0lstm/recurrent_kernel/Assign+lstm/recurrent_kernel/Read/ReadVariableOp:0(2)lstm/recurrent_kernel/Initializer/mul_1:08
d
lstm/bias:0lstm/bias/Assignlstm/bias/Read/ReadVariableOp:0(2lstm/bias/Initializer/concat:08
|
output/kernel:0output/kernel/Assign#output/kernel/Read/ReadVariableOp:0(2*output/kernel/Initializer/random_uniform:08
k
output/bias:0output/bias/Assign!output/bias/Read/ReadVariableOp:0(2output/bias/Initializer/zeros:08"иа
while_contextХаСа
К>
/bidirectional/forward_gru_1/while/while_context  *,bidirectional/forward_gru_1/while/LoopCond:02)bidirectional/forward_gru_1/while/Merge:0:,bidirectional/forward_gru_1/while/Identity:0B(bidirectional/forward_gru_1/while/Exit:0B*bidirectional/forward_gru_1/while/Exit_1:0B*bidirectional/forward_gru_1/while/Exit_2:0J┬:
 bidirectional/forward_gru/bias:0
"bidirectional/forward_gru/kernel:0
,bidirectional/forward_gru/recurrent_kernel:0
)bidirectional/forward_gru_1/TensorArray:0
Xbidirectional/forward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
+bidirectional/forward_gru_1/TensorArray_1:0
'bidirectional/forward_gru_1/ones_like:0
-bidirectional/forward_gru_1/strided_slice_1:0
)bidirectional/forward_gru_1/while/Add_1:0
)bidirectional/forward_gru_1/while/Add_3:0
+bidirectional/forward_gru_1/while/BiasAdd:0
-bidirectional/forward_gru_1/while/BiasAdd_1:0
-bidirectional/forward_gru_1/while/BiasAdd_2:0
)bidirectional/forward_gru_1/while/Const:0
+bidirectional/forward_gru_1/while/Const_1:0
+bidirectional/forward_gru_1/while/Const_2:0
+bidirectional/forward_gru_1/while/Const_3:0
)bidirectional/forward_gru_1/while/Enter:0
+bidirectional/forward_gru_1/while/Enter_1:0
+bidirectional/forward_gru_1/while/Enter_2:0
(bidirectional/forward_gru_1/while/Exit:0
*bidirectional/forward_gru_1/while/Exit_1:0
*bidirectional/forward_gru_1/while/Exit_2:0
,bidirectional/forward_gru_1/while/Identity:0
.bidirectional/forward_gru_1/while/Identity_1:0
.bidirectional/forward_gru_1/while/Identity_2:0
.bidirectional/forward_gru_1/while/Less/Enter:0
(bidirectional/forward_gru_1/while/Less:0
,bidirectional/forward_gru_1/while/LoopCond:0
*bidirectional/forward_gru_1/while/MatMul:0
,bidirectional/forward_gru_1/while/MatMul_1:0
,bidirectional/forward_gru_1/while/MatMul_2:0
,bidirectional/forward_gru_1/while/MatMul_3:0
,bidirectional/forward_gru_1/while/MatMul_4:0
,bidirectional/forward_gru_1/while/MatMul_5:0
)bidirectional/forward_gru_1/while/Merge:0
)bidirectional/forward_gru_1/while/Merge:1
+bidirectional/forward_gru_1/while/Merge_1:0
+bidirectional/forward_gru_1/while/Merge_1:1
+bidirectional/forward_gru_1/while/Merge_2:0
+bidirectional/forward_gru_1/while/Merge_2:1
)bidirectional/forward_gru_1/while/Mul_3:0
)bidirectional/forward_gru_1/while/Mul_4:0
1bidirectional/forward_gru_1/while/NextIteration:0
3bidirectional/forward_gru_1/while/NextIteration_1:0
3bidirectional/forward_gru_1/while/NextIteration_2:0
8bidirectional/forward_gru_1/while/ReadVariableOp/Enter:0
2bidirectional/forward_gru_1/while/ReadVariableOp:0
4bidirectional/forward_gru_1/while/ReadVariableOp_1:0
4bidirectional/forward_gru_1/while/ReadVariableOp_2:0
:bidirectional/forward_gru_1/while/ReadVariableOp_3/Enter:0
4bidirectional/forward_gru_1/while/ReadVariableOp_3:0
4bidirectional/forward_gru_1/while/ReadVariableOp_4:0
4bidirectional/forward_gru_1/while/ReadVariableOp_5:0
:bidirectional/forward_gru_1/while/ReadVariableOp_6/Enter:0
4bidirectional/forward_gru_1/while/ReadVariableOp_6:0
4bidirectional/forward_gru_1/while/ReadVariableOp_7:0
4bidirectional/forward_gru_1/while/ReadVariableOp_8:0
*bidirectional/forward_gru_1/while/Switch:0
*bidirectional/forward_gru_1/while/Switch:1
,bidirectional/forward_gru_1/while/Switch_1:0
,bidirectional/forward_gru_1/while/Switch_1:1
,bidirectional/forward_gru_1/while/Switch_2:0
,bidirectional/forward_gru_1/while/Switch_2:1
(bidirectional/forward_gru_1/while/Tanh:0
;bidirectional/forward_gru_1/while/TensorArrayReadV3/Enter:0
=bidirectional/forward_gru_1/while/TensorArrayReadV3/Enter_1:0
5bidirectional/forward_gru_1/while/TensorArrayReadV3:0
Mbidirectional/forward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Gbidirectional/forward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3:0
'bidirectional/forward_gru_1/while/add:0
)bidirectional/forward_gru_1/while/add_2:0
)bidirectional/forward_gru_1/while/add_4:0
)bidirectional/forward_gru_1/while/add_5:0
+bidirectional/forward_gru_1/while/add_6/y:0
)bidirectional/forward_gru_1/while/add_6:0
;bidirectional/forward_gru_1/while/clip_by_value/Minimum/y:0
9bidirectional/forward_gru_1/while/clip_by_value/Minimum:0
3bidirectional/forward_gru_1/while/clip_by_value/y:0
1bidirectional/forward_gru_1/while/clip_by_value:0
=bidirectional/forward_gru_1/while/clip_by_value_1/Minimum/y:0
;bidirectional/forward_gru_1/while/clip_by_value_1/Minimum:0
5bidirectional/forward_gru_1/while/clip_by_value_1/y:0
3bidirectional/forward_gru_1/while/clip_by_value_1:0
-bidirectional/forward_gru_1/while/mul/Enter:0
'bidirectional/forward_gru_1/while/mul:0
)bidirectional/forward_gru_1/while/mul_1:0
)bidirectional/forward_gru_1/while/mul_2:0
)bidirectional/forward_gru_1/while/mul_5:0
)bidirectional/forward_gru_1/while/mul_6:0
)bidirectional/forward_gru_1/while/mul_7:0
7bidirectional/forward_gru_1/while/strided_slice/stack:0
9bidirectional/forward_gru_1/while/strided_slice/stack_1:0
9bidirectional/forward_gru_1/while/strided_slice/stack_2:0
1bidirectional/forward_gru_1/while/strided_slice:0
9bidirectional/forward_gru_1/while/strided_slice_1/stack:0
;bidirectional/forward_gru_1/while/strided_slice_1/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_1/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_1:0
9bidirectional/forward_gru_1/while/strided_slice_2/stack:0
;bidirectional/forward_gru_1/while/strided_slice_2/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_2/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_2:0
9bidirectional/forward_gru_1/while/strided_slice_3/stack:0
;bidirectional/forward_gru_1/while/strided_slice_3/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_3/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_3:0
9bidirectional/forward_gru_1/while/strided_slice_4/stack:0
;bidirectional/forward_gru_1/while/strided_slice_4/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_4/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_4:0
9bidirectional/forward_gru_1/while/strided_slice_5/stack:0
;bidirectional/forward_gru_1/while/strided_slice_5/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_5/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_5:0
9bidirectional/forward_gru_1/while/strided_slice_6/stack:0
;bidirectional/forward_gru_1/while/strided_slice_6/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_6/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_6:0
9bidirectional/forward_gru_1/while/strided_slice_7/stack:0
;bidirectional/forward_gru_1/while/strided_slice_7/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_7/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_7:0
9bidirectional/forward_gru_1/while/strided_slice_8/stack:0
;bidirectional/forward_gru_1/while/strided_slice_8/stack_1:0
;bidirectional/forward_gru_1/while/strided_slice_8/stack_2:0
3bidirectional/forward_gru_1/while/strided_slice_8:0
)bidirectional/forward_gru_1/while/sub/x:0
'bidirectional/forward_gru_1/while/sub:0X
'bidirectional/forward_gru_1/ones_like:0-bidirectional/forward_gru_1/while/mul/Enter:0j
,bidirectional/forward_gru/recurrent_kernel:0:bidirectional/forward_gru_1/while/ReadVariableOp_6/Enter:0_
-bidirectional/forward_gru_1/strided_slice_1:0.bidirectional/forward_gru_1/while/Less/Enter:0Щ
Xbidirectional/forward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0=bidirectional/forward_gru_1/while/TensorArrayReadV3/Enter_1:0h
)bidirectional/forward_gru_1/TensorArray:0;bidirectional/forward_gru_1/while/TensorArrayReadV3/Enter:0^
"bidirectional/forward_gru/kernel:08bidirectional/forward_gru_1/while/ReadVariableOp/Enter:0|
+bidirectional/forward_gru_1/TensorArray_1:0Mbidirectional/forward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0^
 bidirectional/forward_gru/bias:0:bidirectional/forward_gru_1/while/ReadVariableOp_3/Enter:0R)bidirectional/forward_gru_1/while/Enter:0R+bidirectional/forward_gru_1/while/Enter_1:0R+bidirectional/forward_gru_1/while/Enter_2:0
е?
0bidirectional/backward_gru_1/while/while_context  *-bidirectional/backward_gru_1/while/LoopCond:02*bidirectional/backward_gru_1/while/Merge:0:-bidirectional/backward_gru_1/while/Identity:0B)bidirectional/backward_gru_1/while/Exit:0B+bidirectional/backward_gru_1/while/Exit_1:0B+bidirectional/backward_gru_1/while/Exit_2:0J╙;
!bidirectional/backward_gru/bias:0
#bidirectional/backward_gru/kernel:0
-bidirectional/backward_gru/recurrent_kernel:0
*bidirectional/backward_gru_1/TensorArray:0
Ybidirectional/backward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
,bidirectional/backward_gru_1/TensorArray_1:0
(bidirectional/backward_gru_1/ones_like:0
.bidirectional/backward_gru_1/strided_slice_1:0
*bidirectional/backward_gru_1/while/Add_1:0
*bidirectional/backward_gru_1/while/Add_3:0
,bidirectional/backward_gru_1/while/BiasAdd:0
.bidirectional/backward_gru_1/while/BiasAdd_1:0
.bidirectional/backward_gru_1/while/BiasAdd_2:0
*bidirectional/backward_gru_1/while/Const:0
,bidirectional/backward_gru_1/while/Const_1:0
,bidirectional/backward_gru_1/while/Const_2:0
,bidirectional/backward_gru_1/while/Const_3:0
*bidirectional/backward_gru_1/while/Enter:0
,bidirectional/backward_gru_1/while/Enter_1:0
,bidirectional/backward_gru_1/while/Enter_2:0
)bidirectional/backward_gru_1/while/Exit:0
+bidirectional/backward_gru_1/while/Exit_1:0
+bidirectional/backward_gru_1/while/Exit_2:0
-bidirectional/backward_gru_1/while/Identity:0
/bidirectional/backward_gru_1/while/Identity_1:0
/bidirectional/backward_gru_1/while/Identity_2:0
/bidirectional/backward_gru_1/while/Less/Enter:0
)bidirectional/backward_gru_1/while/Less:0
-bidirectional/backward_gru_1/while/LoopCond:0
+bidirectional/backward_gru_1/while/MatMul:0
-bidirectional/backward_gru_1/while/MatMul_1:0
-bidirectional/backward_gru_1/while/MatMul_2:0
-bidirectional/backward_gru_1/while/MatMul_3:0
-bidirectional/backward_gru_1/while/MatMul_4:0
-bidirectional/backward_gru_1/while/MatMul_5:0
*bidirectional/backward_gru_1/while/Merge:0
*bidirectional/backward_gru_1/while/Merge:1
,bidirectional/backward_gru_1/while/Merge_1:0
,bidirectional/backward_gru_1/while/Merge_1:1
,bidirectional/backward_gru_1/while/Merge_2:0
,bidirectional/backward_gru_1/while/Merge_2:1
*bidirectional/backward_gru_1/while/Mul_3:0
*bidirectional/backward_gru_1/while/Mul_4:0
2bidirectional/backward_gru_1/while/NextIteration:0
4bidirectional/backward_gru_1/while/NextIteration_1:0
4bidirectional/backward_gru_1/while/NextIteration_2:0
9bidirectional/backward_gru_1/while/ReadVariableOp/Enter:0
3bidirectional/backward_gru_1/while/ReadVariableOp:0
5bidirectional/backward_gru_1/while/ReadVariableOp_1:0
5bidirectional/backward_gru_1/while/ReadVariableOp_2:0
;bidirectional/backward_gru_1/while/ReadVariableOp_3/Enter:0
5bidirectional/backward_gru_1/while/ReadVariableOp_3:0
5bidirectional/backward_gru_1/while/ReadVariableOp_4:0
5bidirectional/backward_gru_1/while/ReadVariableOp_5:0
;bidirectional/backward_gru_1/while/ReadVariableOp_6/Enter:0
5bidirectional/backward_gru_1/while/ReadVariableOp_6:0
5bidirectional/backward_gru_1/while/ReadVariableOp_7:0
5bidirectional/backward_gru_1/while/ReadVariableOp_8:0
+bidirectional/backward_gru_1/while/Switch:0
+bidirectional/backward_gru_1/while/Switch:1
-bidirectional/backward_gru_1/while/Switch_1:0
-bidirectional/backward_gru_1/while/Switch_1:1
-bidirectional/backward_gru_1/while/Switch_2:0
-bidirectional/backward_gru_1/while/Switch_2:1
)bidirectional/backward_gru_1/while/Tanh:0
<bidirectional/backward_gru_1/while/TensorArrayReadV3/Enter:0
>bidirectional/backward_gru_1/while/TensorArrayReadV3/Enter_1:0
6bidirectional/backward_gru_1/while/TensorArrayReadV3:0
Nbidirectional/backward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Hbidirectional/backward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3:0
(bidirectional/backward_gru_1/while/add:0
*bidirectional/backward_gru_1/while/add_2:0
*bidirectional/backward_gru_1/while/add_4:0
*bidirectional/backward_gru_1/while/add_5:0
,bidirectional/backward_gru_1/while/add_6/y:0
*bidirectional/backward_gru_1/while/add_6:0
<bidirectional/backward_gru_1/while/clip_by_value/Minimum/y:0
:bidirectional/backward_gru_1/while/clip_by_value/Minimum:0
4bidirectional/backward_gru_1/while/clip_by_value/y:0
2bidirectional/backward_gru_1/while/clip_by_value:0
>bidirectional/backward_gru_1/while/clip_by_value_1/Minimum/y:0
<bidirectional/backward_gru_1/while/clip_by_value_1/Minimum:0
6bidirectional/backward_gru_1/while/clip_by_value_1/y:0
4bidirectional/backward_gru_1/while/clip_by_value_1:0
.bidirectional/backward_gru_1/while/mul/Enter:0
(bidirectional/backward_gru_1/while/mul:0
*bidirectional/backward_gru_1/while/mul_1:0
*bidirectional/backward_gru_1/while/mul_2:0
*bidirectional/backward_gru_1/while/mul_5:0
*bidirectional/backward_gru_1/while/mul_6:0
*bidirectional/backward_gru_1/while/mul_7:0
8bidirectional/backward_gru_1/while/strided_slice/stack:0
:bidirectional/backward_gru_1/while/strided_slice/stack_1:0
:bidirectional/backward_gru_1/while/strided_slice/stack_2:0
2bidirectional/backward_gru_1/while/strided_slice:0
:bidirectional/backward_gru_1/while/strided_slice_1/stack:0
<bidirectional/backward_gru_1/while/strided_slice_1/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_1/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_1:0
:bidirectional/backward_gru_1/while/strided_slice_2/stack:0
<bidirectional/backward_gru_1/while/strided_slice_2/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_2/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_2:0
:bidirectional/backward_gru_1/while/strided_slice_3/stack:0
<bidirectional/backward_gru_1/while/strided_slice_3/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_3/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_3:0
:bidirectional/backward_gru_1/while/strided_slice_4/stack:0
<bidirectional/backward_gru_1/while/strided_slice_4/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_4/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_4:0
:bidirectional/backward_gru_1/while/strided_slice_5/stack:0
<bidirectional/backward_gru_1/while/strided_slice_5/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_5/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_5:0
:bidirectional/backward_gru_1/while/strided_slice_6/stack:0
<bidirectional/backward_gru_1/while/strided_slice_6/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_6/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_6:0
:bidirectional/backward_gru_1/while/strided_slice_7/stack:0
<bidirectional/backward_gru_1/while/strided_slice_7/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_7/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_7:0
:bidirectional/backward_gru_1/while/strided_slice_8/stack:0
<bidirectional/backward_gru_1/while/strided_slice_8/stack_1:0
<bidirectional/backward_gru_1/while/strided_slice_8/stack_2:0
4bidirectional/backward_gru_1/while/strided_slice_8:0
*bidirectional/backward_gru_1/while/sub/x:0
(bidirectional/backward_gru_1/while/sub:0~
,bidirectional/backward_gru_1/TensorArray_1:0Nbidirectional/backward_gru_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0l
-bidirectional/backward_gru/recurrent_kernel:0;bidirectional/backward_gru_1/while/ReadVariableOp_6/Enter:0Ы
Ybidirectional/backward_gru_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0>bidirectional/backward_gru_1/while/TensorArrayReadV3/Enter_1:0Z
(bidirectional/backward_gru_1/ones_like:0.bidirectional/backward_gru_1/while/mul/Enter:0`
#bidirectional/backward_gru/kernel:09bidirectional/backward_gru_1/while/ReadVariableOp/Enter:0a
.bidirectional/backward_gru_1/strided_slice_1:0/bidirectional/backward_gru_1/while/Less/Enter:0`
!bidirectional/backward_gru/bias:0;bidirectional/backward_gru_1/while/ReadVariableOp_3/Enter:0j
*bidirectional/backward_gru_1/TensorArray:0<bidirectional/backward_gru_1/while/TensorArrayReadV3/Enter:0R*bidirectional/backward_gru_1/while/Enter:0R,bidirectional/backward_gru_1/while/Enter_1:0R,bidirectional/backward_gru_1/while/Enter_2:0
┘"
lstm/while/while_context  *lstm/while/LoopCond:02lstm/while/Merge:0:lstm/while/Identity:0Blstm/while/Exit:0Blstm/while/Exit_1:0Blstm/while/Exit_2:0Blstm/while/Exit_3:0J╠ 
lstm/TensorArray:0
Alstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
lstm/TensorArray_1:0
lstm/bias:0
lstm/kernel:0
lstm/ones_like:0
lstm/recurrent_kernel:0
lstm/strided_slice_1:0
lstm/while/Add_1:0
lstm/while/Add_3:0
lstm/while/Add_7:0
lstm/while/BiasAdd:0
lstm/while/BiasAdd_1:0
lstm/while/BiasAdd_2:0
lstm/while/BiasAdd_3:0
lstm/while/Const:0
lstm/while/Const_1:0
lstm/while/Const_2:0
lstm/while/Const_3:0
lstm/while/Const_4:0
lstm/while/Const_5:0
lstm/while/Const_6:0
lstm/while/Const_7:0
lstm/while/Enter:0
lstm/while/Enter_1:0
lstm/while/Enter_2:0
lstm/while/Enter_3:0
lstm/while/Exit:0
lstm/while/Exit_1:0
lstm/while/Exit_2:0
lstm/while/Exit_3:0
lstm/while/Identity:0
lstm/while/Identity_1:0
lstm/while/Identity_2:0
lstm/while/Identity_3:0
lstm/while/Less/Enter:0
lstm/while/Less:0
lstm/while/LoopCond:0
lstm/while/MatMul:0
lstm/while/MatMul_1:0
lstm/while/MatMul_2:0
lstm/while/MatMul_3:0
lstm/while/MatMul_4:0
lstm/while/MatMul_5:0
lstm/while/MatMul_6:0
lstm/while/MatMul_7:0
lstm/while/Merge:0
lstm/while/Merge:1
lstm/while/Merge_1:0
lstm/while/Merge_1:1
lstm/while/Merge_2:0
lstm/while/Merge_2:1
lstm/while/Merge_3:0
lstm/while/Merge_3:1
lstm/while/Mul_4:0
lstm/while/Mul_5:0
lstm/while/Mul_8:0
lstm/while/NextIteration:0
lstm/while/NextIteration_1:0
lstm/while/NextIteration_2:0
lstm/while/NextIteration_3:0
!lstm/while/ReadVariableOp/Enter:0
lstm/while/ReadVariableOp:0
lstm/while/ReadVariableOp_1:0
lstm/while/ReadVariableOp_2:0
lstm/while/ReadVariableOp_3:0
lstm/while/Switch:0
lstm/while/Switch:1
lstm/while/Switch_1:0
lstm/while/Switch_1:1
lstm/while/Switch_2:0
lstm/while/Switch_2:1
lstm/while/Switch_3:0
lstm/while/Switch_3:1
lstm/while/Tanh:0
lstm/while/Tanh_1:0
$lstm/while/TensorArrayReadV3/Enter:0
&lstm/while/TensorArrayReadV3/Enter_1:0
lstm/while/TensorArrayReadV3:0
6lstm/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
0lstm/while/TensorArrayWrite/TensorArrayWriteV3:0
lstm/while/add:0
lstm/while/add_2:0
lstm/while/add_4:0
lstm/while/add_5:0
lstm/while/add_6:0
lstm/while/add_8/y:0
lstm/while/add_8:0
$lstm/while/clip_by_value/Minimum/y:0
"lstm/while/clip_by_value/Minimum:0
lstm/while/clip_by_value/y:0
lstm/while/clip_by_value:0
&lstm/while/clip_by_value_1/Minimum/y:0
$lstm/while/clip_by_value_1/Minimum:0
lstm/while/clip_by_value_1/y:0
lstm/while/clip_by_value_1:0
&lstm/while/clip_by_value_2/Minimum/y:0
$lstm/while/clip_by_value_2/Minimum:0
lstm/while/clip_by_value_2/y:0
lstm/while/clip_by_value_2:0
lstm/while/mul/Enter:0
lstm/while/mul:0
lstm/while/mul_1:0
lstm/while/mul_2:0
lstm/while/mul_3:0
lstm/while/mul_6:0
lstm/while/mul_7:0
lstm/while/mul_9:0
'lstm/while/split/ReadVariableOp/Enter:0
!lstm/while/split/ReadVariableOp:0
lstm/while/split/split_dim:0
lstm/while/split:0
lstm/while/split:1
lstm/while/split:2
lstm/while/split:3
)lstm/while/split_1/ReadVariableOp/Enter:0
#lstm/while/split_1/ReadVariableOp:0
lstm/while/split_1/split_dim:0
lstm/while/split_1:0
lstm/while/split_1:1
lstm/while/split_1:2
lstm/while/split_1:3
 lstm/while/strided_slice/stack:0
"lstm/while/strided_slice/stack_1:0
"lstm/while/strided_slice/stack_2:0
lstm/while/strided_slice:0
"lstm/while/strided_slice_1/stack:0
$lstm/while/strided_slice_1/stack_1:0
$lstm/while/strided_slice_1/stack_2:0
lstm/while/strided_slice_1:0
"lstm/while/strided_slice_2/stack:0
$lstm/while/strided_slice_2/stack_1:0
$lstm/while/strided_slice_2/stack_2:0
lstm/while/strided_slice_2:0
"lstm/while/strided_slice_3/stack:0
$lstm/while/strided_slice_3/stack_1:0
$lstm/while/strided_slice_3/stack_2:0
lstm/while/strided_slice_3:0<
lstm/recurrent_kernel:0!lstm/while/ReadVariableOp/Enter:0:
lstm/TensorArray:0$lstm/while/TensorArrayReadV3/Enter:0N
lstm/TensorArray_1:06lstm/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0k
Alstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0&lstm/while/TensorArrayReadV3/Enter_1:08
lstm/kernel:0'lstm/while/split/ReadVariableOp/Enter:08
lstm/bias:0)lstm/while/split_1/ReadVariableOp/Enter:0*
lstm/ones_like:0lstm/while/mul/Enter:01
lstm/strided_slice_1:0lstm/while/Less/Enter:0Rlstm/while/Enter:0Rlstm/while/Enter_1:0Rlstm/while/Enter_2:0Rlstm/while/Enter_3:0"%
saved_model_main_op


group_deps*Т
serving_default
.
input%
Placeholder:0         Ї1
output'
output/Softmax:0         tensorflow/serving/predict