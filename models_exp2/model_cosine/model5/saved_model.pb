£
®
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ø
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:@*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:@*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:@@*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:@*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@@*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:@*
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:@@*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:@*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:@@*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:@*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:@@*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:@*
dtype0
x
Outputs/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameOutputs/kernel
q
"Outputs/kernel/Read/ReadVariableOpReadVariableOpOutputs/kernel*
_output_shapes

:@*
dtype0
p
Outputs/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutputs/bias
i
 Outputs/bias/Read/ReadVariableOpReadVariableOpOutputs/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_15/kernel/m

*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_16/kernel/m

*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_17/kernel/m

*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_18/kernel/m

*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_19/kernel/m

*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_20/kernel/m

*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:@*
dtype0

Adam/Outputs/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/Outputs/kernel/m

)Adam/Outputs/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Outputs/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/Outputs/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Outputs/bias/m
w
'Adam/Outputs/bias/m/Read/ReadVariableOpReadVariableOpAdam/Outputs/bias/m*
_output_shapes
:*
dtype0

Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_15/kernel/v

*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_16/kernel/v

*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_17/kernel/v

*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_18/kernel/v

*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_19/kernel/v

*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_20/kernel/v

*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:@*
dtype0

Adam/Outputs/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/Outputs/kernel/v

)Adam/Outputs/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Outputs/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/Outputs/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Outputs/bias/v
w
'Adam/Outputs/bias/v/Read/ReadVariableOpReadVariableOpAdam/Outputs/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ÉH
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*H
valueúGB÷G BðG

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
É
9iter

:beta_1

;beta_2
	<decay
=learning_ratemqmrmsmtmumv!mw"mx'my(mz-m{.m|3m}4m~vvvvvv!v"v'v(v-v.v3v4v
 
f
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
f
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
­

regularization_losses
>layer_regularization_losses

?layers
trainable_variables
@non_trainable_variables
Ametrics
	variables
Blayer_metrics
 
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Clayer_regularization_losses

Dlayers
trainable_variables
Enon_trainable_variables
Fmetrics
	variables
Glayer_metrics
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Hlayer_regularization_losses

Ilayers
trainable_variables
Jnon_trainable_variables
Kmetrics
	variables
Llayer_metrics
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Mlayer_regularization_losses

Nlayers
trainable_variables
Onon_trainable_variables
Pmetrics
	variables
Qlayer_metrics
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
­
#regularization_losses
Rlayer_regularization_losses

Slayers
$trainable_variables
Tnon_trainable_variables
Umetrics
%	variables
Vlayer_metrics
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
­
)regularization_losses
Wlayer_regularization_losses

Xlayers
*trainable_variables
Ynon_trainable_variables
Zmetrics
+	variables
[layer_metrics
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
­
/regularization_losses
\layer_regularization_losses

]layers
0trainable_variables
^non_trainable_variables
_metrics
1	variables
`layer_metrics
ZX
VARIABLE_VALUEOutputs/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEOutputs/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
­
5regularization_losses
alayer_regularization_losses

blayers
6trainable_variables
cnon_trainable_variables
dmetrics
7	variables
elayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7
 

f0
g1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	htotal
	icount
j	variables
k	keras_api
D
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

j	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

o	variables
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_17/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_17/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Outputs/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Outputs/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_17/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_17/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Outputs/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Outputs/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_InputsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputsdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasOutputs/kernelOutputs/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_21702
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp"Outputs/kernel/Read/ReadVariableOp Outputs/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp)Adam/Outputs/kernel/m/Read/ReadVariableOp'Adam/Outputs/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp)Adam/Outputs/kernel/v/Read/ReadVariableOp'Adam/Outputs/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_22397


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasOutputs/kernelOutputs/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/mAdam/dense_17/bias/mAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/Outputs/kernel/mAdam/Outputs/bias/mAdam/dense_15/kernel/vAdam/dense_15/bias/vAdam/dense_16/kernel/vAdam/dense_16/bias/vAdam/dense_17/kernel/vAdam/dense_17/bias/vAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/vAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/Outputs/kernel/vAdam/Outputs/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_22560¬
«
¢
__inference_loss_fn_4_22210>
:dense_19_kernel_regularizer_square_readvariableop_resource
identity¢1dense_19/kernel/Regularizer/Square/ReadVariableOpá
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_19_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul
IdentityIdentity#dense_19/kernel/Regularizer/mul:z:02^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp
ê

C__inference_dense_16_layer_call_and_return_conditional_losses_21999

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý	
²
&__inference_cosine_layer_call_fn_21515

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cosine_layer_call_and_return_conditional_losses_214842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ê

C__inference_dense_20_layer_call_and_return_conditional_losses_22127

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_20/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¹	
¯
#__inference_signature_wrapper_21702

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_210662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ÿ_
¢
A__inference_cosine_layer_call_and_return_conditional_losses_21331

inputs
dense_15_21098
dense_15_21100
dense_16_21131
dense_16_21133
dense_17_21164
dense_17_21166
dense_18_21197
dense_18_21199
dense_19_21230
dense_19_21232
dense_20_21263
dense_20_21265
outputs_21289
outputs_21291
identity¢Outputs/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall¢1dense_17/kernel/Regularizer/Square/ReadVariableOp¢ dense_18/StatefulPartitionedCall¢1dense_18/kernel/Regularizer/Square/ReadVariableOp¢ dense_19/StatefulPartitionedCall¢1dense_19/kernel/Regularizer/Square/ReadVariableOp¢ dense_20/StatefulPartitionedCall¢1dense_20/kernel/Regularizer/Square/ReadVariableOp
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_21098dense_15_21100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_210872"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_21131dense_16_21133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_211202"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_21164dense_17_21166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_211532"
 dense_17/StatefulPartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_21197dense_18_21199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_211862"
 dense_18/StatefulPartitionedCall´
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_21230dense_19_21232*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_212192"
 dense_19/StatefulPartitionedCall´
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_21263dense_20_21265*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_212522"
 dense_20/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0outputs_21289outputs_21291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Outputs_layer_call_and_return_conditional_losses_212782!
Outputs/StatefulPartitionedCallµ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_21098*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulµ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_21131*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulµ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_17_21164*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulµ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_21197*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulµ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_21230*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulµ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_20_21263*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mul¨
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall2^dense_17/kernel/Regularizer/Square/ReadVariableOp!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ê

C__inference_dense_17_layer_call_and_return_conditional_losses_21153

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_17/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú
}
(__inference_dense_20_layer_call_fn_22136

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_212522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_16_layer_call_and_return_conditional_losses_21120

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_15_layer_call_and_return_conditional_losses_21967

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
}
æ
A__inference_cosine_layer_call_and_return_conditional_losses_21790

inputs+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource*
&outputs_matmul_readvariableop_resource+
'outputs_biasadd_readvariableop_resource
identity¢Outputs/BiasAdd/ReadVariableOp¢Outputs/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢1dense_17/kernel/Regularizer/Square/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢1dense_18/kernel/Regularizer/Square/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢1dense_19/kernel/Regularizer/Square/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢1dense_20/kernel/Regularizer/Square/ReadVariableOp¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_15/BiasAdd/ReadVariableOp¥
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_15/Relu¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_16/MatMul/ReadVariableOp£
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp¥
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/Relu¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_17/MatMul/ReadVariableOp£
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_17/BiasAdds
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_17/Relu¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_18/MatMul/ReadVariableOp£
dense_18/MatMulMatMuldense_17/Relu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/Relu¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_19/MatMul/ReadVariableOp£
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/Relu¨
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_20/MatMul/ReadVariableOp£
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_20/MatMul§
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_20/BiasAdd/ReadVariableOp¥
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_20/BiasAdds
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_20/Relu¥
Outputs/MatMul/ReadVariableOpReadVariableOp&outputs_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
Outputs/MatMul/ReadVariableOp 
Outputs/MatMulMatMuldense_20/Relu:activations:0%Outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Outputs/MatMul¤
Outputs/BiasAdd/ReadVariableOpReadVariableOp'outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Outputs/BiasAdd/ReadVariableOp¡
Outputs/BiasAddBiasAddOutputs/MatMul:product:0&Outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Outputs/BiasAddÎ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulÎ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulÎ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulÎ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulÎ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulÎ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mul÷
IdentityIdentityOutputs/BiasAdd:output:0^Outputs/BiasAdd/ReadVariableOp^Outputs/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2@
Outputs/BiasAdd/ReadVariableOpOutputs/BiasAdd/ReadVariableOp2>
Outputs/MatMul/ReadVariableOpOutputs/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_17_layer_call_fn_22040

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_211532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«
¢
__inference_loss_fn_0_22166>
:dense_15_kernel_regularizer_square_readvariableop_resource
identity¢1dense_15/kernel/Regularizer/Square/ReadVariableOpá
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_15_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul
IdentityIdentity#dense_15/kernel/Regularizer/mul:z:02^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp
Ø
|
'__inference_Outputs_layer_call_fn_22155

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Outputs_layer_call_and_return_conditional_losses_212782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
õJ
Ñ

 __inference__wrapped_model_21066

inputs2
.cosine_dense_15_matmul_readvariableop_resource3
/cosine_dense_15_biasadd_readvariableop_resource2
.cosine_dense_16_matmul_readvariableop_resource3
/cosine_dense_16_biasadd_readvariableop_resource2
.cosine_dense_17_matmul_readvariableop_resource3
/cosine_dense_17_biasadd_readvariableop_resource2
.cosine_dense_18_matmul_readvariableop_resource3
/cosine_dense_18_biasadd_readvariableop_resource2
.cosine_dense_19_matmul_readvariableop_resource3
/cosine_dense_19_biasadd_readvariableop_resource2
.cosine_dense_20_matmul_readvariableop_resource3
/cosine_dense_20_biasadd_readvariableop_resource1
-cosine_outputs_matmul_readvariableop_resource2
.cosine_outputs_biasadd_readvariableop_resource
identity¢%cosine/Outputs/BiasAdd/ReadVariableOp¢$cosine/Outputs/MatMul/ReadVariableOp¢&cosine/dense_15/BiasAdd/ReadVariableOp¢%cosine/dense_15/MatMul/ReadVariableOp¢&cosine/dense_16/BiasAdd/ReadVariableOp¢%cosine/dense_16/MatMul/ReadVariableOp¢&cosine/dense_17/BiasAdd/ReadVariableOp¢%cosine/dense_17/MatMul/ReadVariableOp¢&cosine/dense_18/BiasAdd/ReadVariableOp¢%cosine/dense_18/MatMul/ReadVariableOp¢&cosine/dense_19/BiasAdd/ReadVariableOp¢%cosine/dense_19/MatMul/ReadVariableOp¢&cosine/dense_20/BiasAdd/ReadVariableOp¢%cosine/dense_20/MatMul/ReadVariableOp½
%cosine/dense_15/MatMul/ReadVariableOpReadVariableOp.cosine_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%cosine/dense_15/MatMul/ReadVariableOp£
cosine/dense_15/MatMulMatMulinputs-cosine/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_15/MatMul¼
&cosine/dense_15/BiasAdd/ReadVariableOpReadVariableOp/cosine_dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cosine/dense_15/BiasAdd/ReadVariableOpÁ
cosine/dense_15/BiasAddBiasAdd cosine/dense_15/MatMul:product:0.cosine/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_15/BiasAdd
cosine/dense_15/ReluRelu cosine/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_15/Relu½
%cosine/dense_16/MatMul/ReadVariableOpReadVariableOp.cosine_dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%cosine/dense_16/MatMul/ReadVariableOp¿
cosine/dense_16/MatMulMatMul"cosine/dense_15/Relu:activations:0-cosine/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_16/MatMul¼
&cosine/dense_16/BiasAdd/ReadVariableOpReadVariableOp/cosine_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cosine/dense_16/BiasAdd/ReadVariableOpÁ
cosine/dense_16/BiasAddBiasAdd cosine/dense_16/MatMul:product:0.cosine/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_16/BiasAdd
cosine/dense_16/ReluRelu cosine/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_16/Relu½
%cosine/dense_17/MatMul/ReadVariableOpReadVariableOp.cosine_dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%cosine/dense_17/MatMul/ReadVariableOp¿
cosine/dense_17/MatMulMatMul"cosine/dense_16/Relu:activations:0-cosine/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_17/MatMul¼
&cosine/dense_17/BiasAdd/ReadVariableOpReadVariableOp/cosine_dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cosine/dense_17/BiasAdd/ReadVariableOpÁ
cosine/dense_17/BiasAddBiasAdd cosine/dense_17/MatMul:product:0.cosine/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_17/BiasAdd
cosine/dense_17/ReluRelu cosine/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_17/Relu½
%cosine/dense_18/MatMul/ReadVariableOpReadVariableOp.cosine_dense_18_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%cosine/dense_18/MatMul/ReadVariableOp¿
cosine/dense_18/MatMulMatMul"cosine/dense_17/Relu:activations:0-cosine/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_18/MatMul¼
&cosine/dense_18/BiasAdd/ReadVariableOpReadVariableOp/cosine_dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cosine/dense_18/BiasAdd/ReadVariableOpÁ
cosine/dense_18/BiasAddBiasAdd cosine/dense_18/MatMul:product:0.cosine/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_18/BiasAdd
cosine/dense_18/ReluRelu cosine/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_18/Relu½
%cosine/dense_19/MatMul/ReadVariableOpReadVariableOp.cosine_dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%cosine/dense_19/MatMul/ReadVariableOp¿
cosine/dense_19/MatMulMatMul"cosine/dense_18/Relu:activations:0-cosine/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_19/MatMul¼
&cosine/dense_19/BiasAdd/ReadVariableOpReadVariableOp/cosine_dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cosine/dense_19/BiasAdd/ReadVariableOpÁ
cosine/dense_19/BiasAddBiasAdd cosine/dense_19/MatMul:product:0.cosine/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_19/BiasAdd
cosine/dense_19/ReluRelu cosine/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_19/Relu½
%cosine/dense_20/MatMul/ReadVariableOpReadVariableOp.cosine_dense_20_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%cosine/dense_20/MatMul/ReadVariableOp¿
cosine/dense_20/MatMulMatMul"cosine/dense_19/Relu:activations:0-cosine/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_20/MatMul¼
&cosine/dense_20/BiasAdd/ReadVariableOpReadVariableOp/cosine_dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cosine/dense_20/BiasAdd/ReadVariableOpÁ
cosine/dense_20/BiasAddBiasAdd cosine/dense_20/MatMul:product:0.cosine/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_20/BiasAdd
cosine/dense_20/ReluRelu cosine/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cosine/dense_20/Reluº
$cosine/Outputs/MatMul/ReadVariableOpReadVariableOp-cosine_outputs_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$cosine/Outputs/MatMul/ReadVariableOp¼
cosine/Outputs/MatMulMatMul"cosine/dense_20/Relu:activations:0,cosine/Outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cosine/Outputs/MatMul¹
%cosine/Outputs/BiasAdd/ReadVariableOpReadVariableOp.cosine_outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%cosine/Outputs/BiasAdd/ReadVariableOp½
cosine/Outputs/BiasAddBiasAddcosine/Outputs/MatMul:product:0-cosine/Outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cosine/Outputs/BiasAdd¨
IdentityIdentitycosine/Outputs/BiasAdd:output:0&^cosine/Outputs/BiasAdd/ReadVariableOp%^cosine/Outputs/MatMul/ReadVariableOp'^cosine/dense_15/BiasAdd/ReadVariableOp&^cosine/dense_15/MatMul/ReadVariableOp'^cosine/dense_16/BiasAdd/ReadVariableOp&^cosine/dense_16/MatMul/ReadVariableOp'^cosine/dense_17/BiasAdd/ReadVariableOp&^cosine/dense_17/MatMul/ReadVariableOp'^cosine/dense_18/BiasAdd/ReadVariableOp&^cosine/dense_18/MatMul/ReadVariableOp'^cosine/dense_19/BiasAdd/ReadVariableOp&^cosine/dense_19/MatMul/ReadVariableOp'^cosine/dense_20/BiasAdd/ReadVariableOp&^cosine/dense_20/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2N
%cosine/Outputs/BiasAdd/ReadVariableOp%cosine/Outputs/BiasAdd/ReadVariableOp2L
$cosine/Outputs/MatMul/ReadVariableOp$cosine/Outputs/MatMul/ReadVariableOp2P
&cosine/dense_15/BiasAdd/ReadVariableOp&cosine/dense_15/BiasAdd/ReadVariableOp2N
%cosine/dense_15/MatMul/ReadVariableOp%cosine/dense_15/MatMul/ReadVariableOp2P
&cosine/dense_16/BiasAdd/ReadVariableOp&cosine/dense_16/BiasAdd/ReadVariableOp2N
%cosine/dense_16/MatMul/ReadVariableOp%cosine/dense_16/MatMul/ReadVariableOp2P
&cosine/dense_17/BiasAdd/ReadVariableOp&cosine/dense_17/BiasAdd/ReadVariableOp2N
%cosine/dense_17/MatMul/ReadVariableOp%cosine/dense_17/MatMul/ReadVariableOp2P
&cosine/dense_18/BiasAdd/ReadVariableOp&cosine/dense_18/BiasAdd/ReadVariableOp2N
%cosine/dense_18/MatMul/ReadVariableOp%cosine/dense_18/MatMul/ReadVariableOp2P
&cosine/dense_19/BiasAdd/ReadVariableOp&cosine/dense_19/BiasAdd/ReadVariableOp2N
%cosine/dense_19/MatMul/ReadVariableOp%cosine/dense_19/MatMul/ReadVariableOp2P
&cosine/dense_20/BiasAdd/ReadVariableOp&cosine/dense_20/BiasAdd/ReadVariableOp2N
%cosine/dense_20/MatMul/ReadVariableOp%cosine/dense_20/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
Ú
}
(__inference_dense_19_layer_call_fn_22104

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_212192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_15_layer_call_and_return_conditional_losses_21087

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¢
__inference_loss_fn_1_22177>
:dense_16_kernel_regularizer_square_readvariableop_resource
identity¢1dense_16/kernel/Regularizer/Square/ReadVariableOpá
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_16_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul
IdentityIdentity#dense_16/kernel/Regularizer/mul:z:02^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp
	
Û
B__inference_Outputs_layer_call_and_return_conditional_losses_22146

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú
}
(__inference_dense_16_layer_call_fn_22008

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_211202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_17_layer_call_and_return_conditional_losses_22031

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_17/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý	
²
&__inference_cosine_layer_call_fn_21911

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cosine_layer_call_and_return_conditional_losses_214842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¢
__inference_loss_fn_2_22188>
:dense_17_kernel_regularizer_square_readvariableop_resource
identity¢1dense_17/kernel/Regularizer/Square/ReadVariableOpá
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_17_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul
IdentityIdentity#dense_17/kernel/Regularizer/mul:z:02^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp
Ö

!__inference__traced_restore_22560
file_prefix$
 assignvariableop_dense_15_kernel$
 assignvariableop_1_dense_15_bias&
"assignvariableop_2_dense_16_kernel$
 assignvariableop_3_dense_16_bias&
"assignvariableop_4_dense_17_kernel$
 assignvariableop_5_dense_17_bias&
"assignvariableop_6_dense_18_kernel$
 assignvariableop_7_dense_18_bias&
"assignvariableop_8_dense_19_kernel$
 assignvariableop_9_dense_19_bias'
#assignvariableop_10_dense_20_kernel%
!assignvariableop_11_dense_20_bias&
"assignvariableop_12_outputs_kernel$
 assignvariableop_13_outputs_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_1.
*assignvariableop_23_adam_dense_15_kernel_m,
(assignvariableop_24_adam_dense_15_bias_m.
*assignvariableop_25_adam_dense_16_kernel_m,
(assignvariableop_26_adam_dense_16_bias_m.
*assignvariableop_27_adam_dense_17_kernel_m,
(assignvariableop_28_adam_dense_17_bias_m.
*assignvariableop_29_adam_dense_18_kernel_m,
(assignvariableop_30_adam_dense_18_bias_m.
*assignvariableop_31_adam_dense_19_kernel_m,
(assignvariableop_32_adam_dense_19_bias_m.
*assignvariableop_33_adam_dense_20_kernel_m,
(assignvariableop_34_adam_dense_20_bias_m-
)assignvariableop_35_adam_outputs_kernel_m+
'assignvariableop_36_adam_outputs_bias_m.
*assignvariableop_37_adam_dense_15_kernel_v,
(assignvariableop_38_adam_dense_15_bias_v.
*assignvariableop_39_adam_dense_16_kernel_v,
(assignvariableop_40_adam_dense_16_bias_v.
*assignvariableop_41_adam_dense_17_kernel_v,
(assignvariableop_42_adam_dense_17_bias_v.
*assignvariableop_43_adam_dense_18_kernel_v,
(assignvariableop_44_adam_dense_18_bias_v.
*assignvariableop_45_adam_dense_19_kernel_v,
(assignvariableop_46_adam_dense_19_bias_v.
*assignvariableop_47_adam_dense_20_kernel_v,
(assignvariableop_48_adam_dense_20_bias_v-
)assignvariableop_49_adam_outputs_kernel_v+
'assignvariableop_50_adam_outputs_bias_v
identity_52¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ú
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueüBù4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesö
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices²
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*æ
_output_shapesÓ
Ð::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_18_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_18_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_19_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_19_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_20_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_20_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ª
AssignVariableOp_12AssignVariableOp"assignvariableop_12_outputs_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¨
AssignVariableOp_13AssignVariableOp assignvariableop_13_outputs_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14¥
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15§
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16§
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¦
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22£
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23²
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_15_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_15_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_16_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_16_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_17_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_17_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_18_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_18_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_19_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_19_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_20_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_20_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35±
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_outputs_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¯
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_outputs_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_15_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_15_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_16_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_16_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41²
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_17_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42°
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_17_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_18_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_18_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45²
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_19_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46°
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_19_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47²
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_20_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48°
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_20_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49±
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_outputs_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¯
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_outputs_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÀ	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51³	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*ã
_input_shapesÑ
Î: :::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
}
æ
A__inference_cosine_layer_call_and_return_conditional_losses_21878

inputs+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource*
&outputs_matmul_readvariableop_resource+
'outputs_biasadd_readvariableop_resource
identity¢Outputs/BiasAdd/ReadVariableOp¢Outputs/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢1dense_17/kernel/Regularizer/Square/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢1dense_18/kernel/Regularizer/Square/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢1dense_19/kernel/Regularizer/Square/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢1dense_20/kernel/Regularizer/Square/ReadVariableOp¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_15/BiasAdd/ReadVariableOp¥
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_15/Relu¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_16/MatMul/ReadVariableOp£
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp¥
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/Relu¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_17/MatMul/ReadVariableOp£
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_17/BiasAdds
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_17/Relu¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_18/MatMul/ReadVariableOp£
dense_18/MatMulMatMuldense_17/Relu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/Relu¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_19/MatMul/ReadVariableOp£
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/Relu¨
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_20/MatMul/ReadVariableOp£
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_20/MatMul§
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_20/BiasAdd/ReadVariableOp¥
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_20/BiasAdds
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_20/Relu¥
Outputs/MatMul/ReadVariableOpReadVariableOp&outputs_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
Outputs/MatMul/ReadVariableOp 
Outputs/MatMulMatMuldense_20/Relu:activations:0%Outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Outputs/MatMul¤
Outputs/BiasAdd/ReadVariableOpReadVariableOp'outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Outputs/BiasAdd/ReadVariableOp¡
Outputs/BiasAddBiasAddOutputs/MatMul:product:0&Outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Outputs/BiasAddÎ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulÎ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulÎ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulÎ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulÎ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulÎ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mul÷
IdentityIdentityOutputs/BiasAdd:output:0^Outputs/BiasAdd/ReadVariableOp^Outputs/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2@
Outputs/BiasAdd/ReadVariableOpOutputs/BiasAdd/ReadVariableOp2>
Outputs/MatMul/ReadVariableOpOutputs/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ_
¢
A__inference_cosine_layer_call_and_return_conditional_losses_21592

inputs
dense_15_21520
dense_15_21522
dense_16_21525
dense_16_21527
dense_17_21530
dense_17_21532
dense_18_21535
dense_18_21537
dense_19_21540
dense_19_21542
dense_20_21545
dense_20_21547
outputs_21550
outputs_21552
identity¢Outputs/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall¢1dense_17/kernel/Regularizer/Square/ReadVariableOp¢ dense_18/StatefulPartitionedCall¢1dense_18/kernel/Regularizer/Square/ReadVariableOp¢ dense_19/StatefulPartitionedCall¢1dense_19/kernel/Regularizer/Square/ReadVariableOp¢ dense_20/StatefulPartitionedCall¢1dense_20/kernel/Regularizer/Square/ReadVariableOp
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_21520dense_15_21522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_210872"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_21525dense_16_21527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_211202"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_21530dense_17_21532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_211532"
 dense_17/StatefulPartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_21535dense_18_21537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_211862"
 dense_18/StatefulPartitionedCall´
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_21540dense_19_21542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_212192"
 dense_19/StatefulPartitionedCall´
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_21545dense_20_21547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_212522"
 dense_20/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0outputs_21550outputs_21552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Outputs_layer_call_and_return_conditional_losses_212782!
Outputs/StatefulPartitionedCallµ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_21520*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulµ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_21525*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulµ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_17_21530*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulµ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_21535*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulµ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_21540*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulµ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_20_21545*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mul¨
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall2^dense_17/kernel/Regularizer/Square/ReadVariableOp!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ_
¢
A__inference_cosine_layer_call_and_return_conditional_losses_21484

inputs
dense_15_21412
dense_15_21414
dense_16_21417
dense_16_21419
dense_17_21422
dense_17_21424
dense_18_21427
dense_18_21429
dense_19_21432
dense_19_21434
dense_20_21437
dense_20_21439
outputs_21442
outputs_21444
identity¢Outputs/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall¢1dense_17/kernel/Regularizer/Square/ReadVariableOp¢ dense_18/StatefulPartitionedCall¢1dense_18/kernel/Regularizer/Square/ReadVariableOp¢ dense_19/StatefulPartitionedCall¢1dense_19/kernel/Regularizer/Square/ReadVariableOp¢ dense_20/StatefulPartitionedCall¢1dense_20/kernel/Regularizer/Square/ReadVariableOp
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_21412dense_15_21414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_210872"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_21417dense_16_21419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_211202"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_21422dense_17_21424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_211532"
 dense_17/StatefulPartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_21427dense_18_21429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_211862"
 dense_18/StatefulPartitionedCall´
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_21432dense_19_21434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_212192"
 dense_19/StatefulPartitionedCall´
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_21437dense_20_21439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_212522"
 dense_20/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0outputs_21442outputs_21444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Outputs_layer_call_and_return_conditional_losses_212782!
Outputs/StatefulPartitionedCallµ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_21412*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulµ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_21417*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulµ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_17_21422*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulµ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_21427*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulµ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_21432*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulµ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_20_21437*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mul¨
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall2^dense_17/kernel/Regularizer/Square/ReadVariableOp!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¢
__inference_loss_fn_5_22221>
:dense_20_kernel_regularizer_square_readvariableop_resource
identity¢1dense_20/kernel/Regularizer/Square/ReadVariableOpá
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_20_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mul
IdentityIdentity#dense_20/kernel/Regularizer/mul:z:02^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp
	
Û
B__inference_Outputs_layer_call_and_return_conditional_losses_21278

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_19_layer_call_and_return_conditional_losses_21219

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_19/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_20_layer_call_and_return_conditional_losses_21252

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_20/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ_
¢
A__inference_cosine_layer_call_and_return_conditional_losses_21406

inputs
dense_15_21334
dense_15_21336
dense_16_21339
dense_16_21341
dense_17_21344
dense_17_21346
dense_18_21349
dense_18_21351
dense_19_21354
dense_19_21356
dense_20_21359
dense_20_21361
outputs_21364
outputs_21366
identity¢Outputs/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall¢1dense_17/kernel/Regularizer/Square/ReadVariableOp¢ dense_18/StatefulPartitionedCall¢1dense_18/kernel/Regularizer/Square/ReadVariableOp¢ dense_19/StatefulPartitionedCall¢1dense_19/kernel/Regularizer/Square/ReadVariableOp¢ dense_20/StatefulPartitionedCall¢1dense_20/kernel/Regularizer/Square/ReadVariableOp
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_21334dense_15_21336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_210872"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_21339dense_16_21341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_211202"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_21344dense_17_21346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_211532"
 dense_17/StatefulPartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_21349dense_18_21351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_211862"
 dense_18/StatefulPartitionedCall´
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_21354dense_19_21356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_212192"
 dense_19/StatefulPartitionedCall´
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_21359dense_20_21361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_212522"
 dense_20/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0outputs_21364outputs_21366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Outputs_layer_call_and_return_conditional_losses_212782!
Outputs/StatefulPartitionedCallµ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_21334*
_output_shapes

:@*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¶
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulµ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_21339*
_output_shapes

:@@*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¶
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulµ
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_17_21344*
_output_shapes

:@@*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp¶
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_17/kernel/Regularizer/Square
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const¾
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_17/kernel/Regularizer/mul/xÀ
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mulµ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_21349*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulµ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_21354*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulµ
1dense_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_20_21359*
_output_shapes

:@@*
dtype023
1dense_20/kernel/Regularizer/Square/ReadVariableOp¶
"dense_20/kernel/Regularizer/SquareSquare9dense_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_20/kernel/Regularizer/Square
!dense_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_20/kernel/Regularizer/Const¾
dense_20/kernel/Regularizer/SumSum&dense_20/kernel/Regularizer/Square:y:0*dense_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/Sum
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_20/kernel/Regularizer/mul/xÀ
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0(dense_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_20/kernel/Regularizer/mul¨
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall2^dense_17/kernel/Regularizer/Square/ReadVariableOp!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall2^dense_20/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2f
1dense_20/kernel/Regularizer/Square/ReadVariableOp1dense_20/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ê

C__inference_dense_18_layer_call_and_return_conditional_losses_22063

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_18/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_18_layer_call_and_return_conditional_losses_21186

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_18/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³f
Û
__inference__traced_save_22397
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop-
)savev2_outputs_kernel_read_readvariableop+
'savev2_outputs_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop4
0savev2_adam_outputs_kernel_m_read_readvariableop2
.savev2_adam_outputs_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop4
0savev2_adam_outputs_kernel_v_read_readvariableop2
.savev2_adam_outputs_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameô
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueüBù4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesð
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop)savev2_outputs_kernel_read_readvariableop'savev2_outputs_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop0savev2_adam_outputs_kernel_m_read_readvariableop.savev2_adam_outputs_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop0savev2_adam_outputs_kernel_v_read_readvariableop.savev2_adam_outputs_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*û
_input_shapesé
æ: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@:: : : : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$  

_output_shapes

:@@: !

_output_shapes
:@:$" 

_output_shapes

:@@: #

_output_shapes
:@:$$ 

_output_shapes

:@: %

_output_shapes
::$& 

_output_shapes

:@: '

_output_shapes
:@:$( 

_output_shapes

:@@: )

_output_shapes
:@:$* 

_output_shapes

:@@: +

_output_shapes
:@:$, 

_output_shapes

:@@: -

_output_shapes
:@:$. 

_output_shapes

:@@: /

_output_shapes
:@:$0 

_output_shapes

:@@: 1

_output_shapes
:@:$2 

_output_shapes

:@: 3

_output_shapes
::4

_output_shapes
: 
«
¢
__inference_loss_fn_3_22199>
:dense_18_kernel_regularizer_square_readvariableop_resource
identity¢1dense_18/kernel/Regularizer/Square/ReadVariableOpá
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_18_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp¶
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_18/kernel/Regularizer/Square
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const¾
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_18/kernel/Regularizer/mul/xÀ
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul
IdentityIdentity#dense_18/kernel/Regularizer/mul:z:02^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp
Ú
}
(__inference_dense_15_layer_call_fn_21976

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_210872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_18_layer_call_fn_22072

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_211862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_19_layer_call_and_return_conditional_losses_22095

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_19/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÅ
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp¶
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_19/kernel/Regularizer/Square
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const¾
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_19/kernel/Regularizer/mul/xÀ
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý	
²
&__inference_cosine_layer_call_fn_21944

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cosine_layer_call_and_return_conditional_losses_215922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý	
²
&__inference_cosine_layer_call_fn_21623

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cosine_layer_call_and_return_conditional_losses_215922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¨
serving_default
9
Inputs/
serving_default_Inputs:0ÿÿÿÿÿÿÿÿÿ;
Outputs0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
áL
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"õH
_tf_keras_networkÙH{"class_name": "Functional", "name": "cosine", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cosine", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["Inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Outputs", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Outputs", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Outputs", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "cosine", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["Inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Outputs", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Outputs", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Outputs", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "Inputs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}}
ª

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layeré{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
¬

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¬

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¬

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¬

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¬

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ó

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+&call_and_return_all_conditional_losses
__call__"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "Outputs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Outputs", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
Ü
9iter

:beta_1

;beta_2
	<decay
=learning_ratemqmrmsmtmumv!mw"mx'my(mz-m{.m|3m}4m~vvvvvv!v"v'v(v-v.v3v4v"
	optimizer
P
0
1
 2
¡3
¢4
£5"
trackable_list_wrapper

0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413"
trackable_list_wrapper

0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413"
trackable_list_wrapper
Î

regularization_losses
>layer_regularization_losses

?layers
trainable_variables
@non_trainable_variables
Ametrics
	variables
Blayer_metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¤serving_default"
signature_map
!:@2dense_15/kernel
:@2dense_15/bias
(
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Clayer_regularization_losses

Dlayers
trainable_variables
Enon_trainable_variables
Fmetrics
	variables
Glayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_16/kernel
:@2dense_16/bias
(
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Hlayer_regularization_losses

Ilayers
trainable_variables
Jnon_trainable_variables
Kmetrics
	variables
Llayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_17/kernel
:@2dense_17/bias
(
 0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Mlayer_regularization_losses

Nlayers
trainable_variables
Onon_trainable_variables
Pmetrics
	variables
Qlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_18/kernel
:@2dense_18/bias
(
¡0"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
°
#regularization_losses
Rlayer_regularization_losses

Slayers
$trainable_variables
Tnon_trainable_variables
Umetrics
%	variables
Vlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_19/kernel
:@2dense_19/bias
(
¢0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
°
)regularization_losses
Wlayer_regularization_losses

Xlayers
*trainable_variables
Ynon_trainable_variables
Zmetrics
+	variables
[layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_20/kernel
:@2dense_20/bias
(
£0"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
°
/regularization_losses
\layer_regularization_losses

]layers
0trainable_variables
^non_trainable_variables
_metrics
1	variables
`layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :@2Outputs/kernel
:2Outputs/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
°
5regularization_losses
alayer_regularization_losses

blayers
6trainable_variables
cnon_trainable_variables
dmetrics
7	variables
elayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
¡0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
¢0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
£0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	htotal
	icount
j	variables
k	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api"Ê
_tf_keras_metric¯{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
h0
i1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
&:$@2Adam/dense_15/kernel/m
 :@2Adam/dense_15/bias/m
&:$@@2Adam/dense_16/kernel/m
 :@2Adam/dense_16/bias/m
&:$@@2Adam/dense_17/kernel/m
 :@2Adam/dense_17/bias/m
&:$@@2Adam/dense_18/kernel/m
 :@2Adam/dense_18/bias/m
&:$@@2Adam/dense_19/kernel/m
 :@2Adam/dense_19/bias/m
&:$@@2Adam/dense_20/kernel/m
 :@2Adam/dense_20/bias/m
%:#@2Adam/Outputs/kernel/m
:2Adam/Outputs/bias/m
&:$@2Adam/dense_15/kernel/v
 :@2Adam/dense_15/bias/v
&:$@@2Adam/dense_16/kernel/v
 :@2Adam/dense_16/bias/v
&:$@@2Adam/dense_17/kernel/v
 :@2Adam/dense_17/bias/v
&:$@@2Adam/dense_18/kernel/v
 :@2Adam/dense_18/bias/v
&:$@@2Adam/dense_19/kernel/v
 :@2Adam/dense_19/bias/v
&:$@@2Adam/dense_20/kernel/v
 :@2Adam/dense_20/bias/v
%:#@2Adam/Outputs/kernel/v
:2Adam/Outputs/bias/v
Ò2Ï
A__inference_cosine_layer_call_and_return_conditional_losses_21878
A__inference_cosine_layer_call_and_return_conditional_losses_21790
A__inference_cosine_layer_call_and_return_conditional_losses_21406
A__inference_cosine_layer_call_and_return_conditional_losses_21331À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
&__inference_cosine_layer_call_fn_21623
&__inference_cosine_layer_call_fn_21911
&__inference_cosine_layer_call_fn_21944
&__inference_cosine_layer_call_fn_21515À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ý2Ú
 __inference__wrapped_model_21066µ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *%¢"
 
Inputsÿÿÿÿÿÿÿÿÿ
í2ê
C__inference_dense_15_layer_call_and_return_conditional_losses_21967¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_15_layer_call_fn_21976¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_16_layer_call_and_return_conditional_losses_21999¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_16_layer_call_fn_22008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_17_layer_call_and_return_conditional_losses_22031¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_17_layer_call_fn_22040¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_18_layer_call_and_return_conditional_losses_22063¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_18_layer_call_fn_22072¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_19_layer_call_and_return_conditional_losses_22095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_19_layer_call_fn_22104¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_20_layer_call_and_return_conditional_losses_22127¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_20_layer_call_fn_22136¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_Outputs_layer_call_and_return_conditional_losses_22146¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_Outputs_layer_call_fn_22155¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
²2¯
__inference_loss_fn_0_22166
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_1_22177
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_2_22188
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_3_22199
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_4_22210
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_5_22221
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ÉBÆ
#__inference_signature_wrapper_21702Inputs"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¢
B__inference_Outputs_layer_call_and_return_conditional_losses_22146\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_Outputs_layer_call_fn_22155O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
 __inference__wrapped_model_21066t!"'(-.34/¢,
%¢"
 
Inputsÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
Outputs!
Outputsÿÿÿÿÿÿÿÿÿµ
A__inference_cosine_layer_call_and_return_conditional_losses_21331p!"'(-.347¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
A__inference_cosine_layer_call_and_return_conditional_losses_21406p!"'(-.347¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
A__inference_cosine_layer_call_and_return_conditional_losses_21790p!"'(-.347¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
A__inference_cosine_layer_call_and_return_conditional_losses_21878p!"'(-.347¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_cosine_layer_call_fn_21515c!"'(-.347¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_cosine_layer_call_fn_21623c!"'(-.347¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_cosine_layer_call_fn_21911c!"'(-.347¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_cosine_layer_call_fn_21944c!"'(-.347¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_15_layer_call_and_return_conditional_losses_21967\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_15_layer_call_fn_21976O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_16_layer_call_and_return_conditional_losses_21999\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_16_layer_call_fn_22008O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_17_layer_call_and_return_conditional_losses_22031\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_17_layer_call_fn_22040O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_18_layer_call_and_return_conditional_losses_22063\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_18_layer_call_fn_22072O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_19_layer_call_and_return_conditional_losses_22095\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_19_layer_call_fn_22104O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_20_layer_call_and_return_conditional_losses_22127\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_20_layer_call_fn_22136O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@:
__inference_loss_fn_0_22166¢

¢ 
ª " :
__inference_loss_fn_1_22177¢

¢ 
ª " :
__inference_loss_fn_2_22188¢

¢ 
ª " :
__inference_loss_fn_3_22199!¢

¢ 
ª " :
__inference_loss_fn_4_22210'¢

¢ 
ª " :
__inference_loss_fn_5_22221-¢

¢ 
ª " ¥
#__inference_signature_wrapper_21702~!"'(-.349¢6
¢ 
/ª,
*
Inputs 
Inputsÿÿÿÿÿÿÿÿÿ"1ª.
,
Outputs!
Outputsÿÿÿÿÿÿÿÿÿ