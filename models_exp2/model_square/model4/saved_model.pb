Ë
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8á
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:@*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:@*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:@@*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:@*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:@@*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:@*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:@@*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
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
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_10/kernel/m

*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_11/kernel/m

*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_14/kernel/m

*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
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
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_10/kernel/v

*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_11/kernel/v

*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_14/kernel/v

*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
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
Ö?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? Bý>
è
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
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
h

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api

2iter

3beta_1

4beta_2
	5decay
6learning_ratememfmgmhmimj mk!ml&mm'mn,mo-mpvqvrvsvtvuvv vw!vx&vy'vz,v{-v|
V
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11
V
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11
 
­
		variables

trainable_variables
7metrics
regularization_losses
8layer_metrics
9layer_regularization_losses

:layers
;non_trainable_variables
 
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
<layer_metrics
=metrics
regularization_losses
trainable_variables
>layer_regularization_losses

?layers
@non_trainable_variables
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
Alayer_metrics
Bmetrics
regularization_losses
trainable_variables
Clayer_regularization_losses

Dlayers
Enon_trainable_variables
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
Flayer_metrics
Gmetrics
regularization_losses
trainable_variables
Hlayer_regularization_losses

Ilayers
Jnon_trainable_variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
­
"	variables
Klayer_metrics
Lmetrics
#regularization_losses
$trainable_variables
Mlayer_regularization_losses

Nlayers
Onon_trainable_variables
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
­
(	variables
Player_metrics
Qmetrics
)regularization_losses
*trainable_variables
Rlayer_regularization_losses

Slayers
Tnon_trainable_variables
ZX
VARIABLE_VALUEOutputs/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEOutputs/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
­
.	variables
Ulayer_metrics
Vmetrics
/regularization_losses
0trainable_variables
Wlayer_regularization_losses

Xlayers
Ynon_trainable_variables
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

Z0
[1
 
 
1
0
1
2
3
4
5
6
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
	\total
	]count
^	variables
_	keras_api
D
	`total
	acount
b
_fn_kwargs
c	variables
d	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

^	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1

c	variables
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Outputs/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Outputs/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Outputs/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Outputs/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_InputsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputsdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasOutputs/kernelOutputs/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_17166
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp"Outputs/kernel/Read/ReadVariableOp Outputs/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp)Adam/Outputs/kernel/m/Read/ReadVariableOp'Adam/Outputs/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp)Adam/Outputs/kernel/v/Read/ReadVariableOp'Adam/Outputs/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
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
__inference__traced_save_17766
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasOutputs/kernelOutputs/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/Outputs/kernel/mAdam/Outputs/bias/mAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/Outputs/kernel/vAdam/Outputs/bias/v*9
Tin2
02.*
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
!__inference__traced_restore_17911ôß	
«
¢
__inference_loss_fn_1_17575>
:dense_11_kernel_regularizer_square_readvariableop_resource
identity¢1dense_11/kernel/Regularizer/Square/ReadVariableOpá
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_11_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mul
IdentityIdentity#dense_11/kernel/Regularizer/mul:z:02^dense_11/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp
Q
£
A__inference_square_layer_call_and_return_conditional_losses_17070

inputs
dense_10_17009
dense_10_17011
dense_11_17014
dense_11_17016
dense_12_17019
dense_12_17021
dense_13_17024
dense_13_17026
dense_14_17029
dense_14_17031
outputs_17034
outputs_17036
identity¢Outputs/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢1dense_10/kernel/Regularizer/Square/ReadVariableOp¢ dense_11/StatefulPartitionedCall¢1dense_11/kernel/Regularizer/Square/ReadVariableOp¢ dense_12/StatefulPartitionedCall¢1dense_12/kernel/Regularizer/Square/ReadVariableOp¢ dense_13/StatefulPartitionedCall¢1dense_13/kernel/Regularizer/Square/ReadVariableOp¢ dense_14/StatefulPartitionedCall¢1dense_14/kernel/Regularizer/Square/ReadVariableOp
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_17009dense_10_17011*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_166412"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_17014dense_11_17016*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_166742"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_17019dense_12_17021*
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
C__inference_dense_12_layer_call_and_return_conditional_losses_167072"
 dense_12/StatefulPartitionedCall´
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_17024dense_13_17026*
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
C__inference_dense_13_layer_call_and_return_conditional_losses_167402"
 dense_13/StatefulPartitionedCall´
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_17029dense_14_17031*
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
C__inference_dense_14_layer_call_and_return_conditional_losses_167732"
 dense_14/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0outputs_17034outputs_17036*
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
B__inference_Outputs_layer_call_and_return_conditional_losses_167992!
Outputs/StatefulPartitionedCallµ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_17009*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulµ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_17014*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulµ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_17019*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulµ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_17024*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulµ
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_14_17029*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulÑ
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall2^dense_11/kernel/Regularizer/Square/ReadVariableOp!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

C__inference_dense_12_layer_call_and_return_conditional_losses_16707

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_12/kernel/Regularizer/Square/ReadVariableOp
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
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«
¢
__inference_loss_fn_3_17597>
:dense_13_kernel_regularizer_square_readvariableop_resource
identity¢1dense_13/kernel/Regularizer/Square/ReadVariableOpá
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_13_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul
IdentityIdentity#dense_13/kernel/Regularizer/mul:z:02^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp
°@
	
 __inference__wrapped_model_16620

inputs2
.square_dense_10_matmul_readvariableop_resource3
/square_dense_10_biasadd_readvariableop_resource2
.square_dense_11_matmul_readvariableop_resource3
/square_dense_11_biasadd_readvariableop_resource2
.square_dense_12_matmul_readvariableop_resource3
/square_dense_12_biasadd_readvariableop_resource2
.square_dense_13_matmul_readvariableop_resource3
/square_dense_13_biasadd_readvariableop_resource2
.square_dense_14_matmul_readvariableop_resource3
/square_dense_14_biasadd_readvariableop_resource1
-square_outputs_matmul_readvariableop_resource2
.square_outputs_biasadd_readvariableop_resource
identity¢%square/Outputs/BiasAdd/ReadVariableOp¢$square/Outputs/MatMul/ReadVariableOp¢&square/dense_10/BiasAdd/ReadVariableOp¢%square/dense_10/MatMul/ReadVariableOp¢&square/dense_11/BiasAdd/ReadVariableOp¢%square/dense_11/MatMul/ReadVariableOp¢&square/dense_12/BiasAdd/ReadVariableOp¢%square/dense_12/MatMul/ReadVariableOp¢&square/dense_13/BiasAdd/ReadVariableOp¢%square/dense_13/MatMul/ReadVariableOp¢&square/dense_14/BiasAdd/ReadVariableOp¢%square/dense_14/MatMul/ReadVariableOp½
%square/dense_10/MatMul/ReadVariableOpReadVariableOp.square_dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%square/dense_10/MatMul/ReadVariableOp£
square/dense_10/MatMulMatMulinputs-square/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_10/MatMul¼
&square/dense_10/BiasAdd/ReadVariableOpReadVariableOp/square_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&square/dense_10/BiasAdd/ReadVariableOpÁ
square/dense_10/BiasAddBiasAdd square/dense_10/MatMul:product:0.square/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_10/BiasAdd
square/dense_10/ReluRelu square/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_10/Relu½
%square/dense_11/MatMul/ReadVariableOpReadVariableOp.square_dense_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%square/dense_11/MatMul/ReadVariableOp¿
square/dense_11/MatMulMatMul"square/dense_10/Relu:activations:0-square/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_11/MatMul¼
&square/dense_11/BiasAdd/ReadVariableOpReadVariableOp/square_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&square/dense_11/BiasAdd/ReadVariableOpÁ
square/dense_11/BiasAddBiasAdd square/dense_11/MatMul:product:0.square/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_11/BiasAdd
square/dense_11/ReluRelu square/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_11/Relu½
%square/dense_12/MatMul/ReadVariableOpReadVariableOp.square_dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%square/dense_12/MatMul/ReadVariableOp¿
square/dense_12/MatMulMatMul"square/dense_11/Relu:activations:0-square/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_12/MatMul¼
&square/dense_12/BiasAdd/ReadVariableOpReadVariableOp/square_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&square/dense_12/BiasAdd/ReadVariableOpÁ
square/dense_12/BiasAddBiasAdd square/dense_12/MatMul:product:0.square/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_12/BiasAdd
square/dense_12/ReluRelu square/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_12/Relu½
%square/dense_13/MatMul/ReadVariableOpReadVariableOp.square_dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%square/dense_13/MatMul/ReadVariableOp¿
square/dense_13/MatMulMatMul"square/dense_12/Relu:activations:0-square/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_13/MatMul¼
&square/dense_13/BiasAdd/ReadVariableOpReadVariableOp/square_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&square/dense_13/BiasAdd/ReadVariableOpÁ
square/dense_13/BiasAddBiasAdd square/dense_13/MatMul:product:0.square/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_13/BiasAdd
square/dense_13/ReluRelu square/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_13/Relu½
%square/dense_14/MatMul/ReadVariableOpReadVariableOp.square_dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%square/dense_14/MatMul/ReadVariableOp¿
square/dense_14/MatMulMatMul"square/dense_13/Relu:activations:0-square/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_14/MatMul¼
&square/dense_14/BiasAdd/ReadVariableOpReadVariableOp/square_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&square/dense_14/BiasAdd/ReadVariableOpÁ
square/dense_14/BiasAddBiasAdd square/dense_14/MatMul:product:0.square/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_14/BiasAdd
square/dense_14/ReluRelu square/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
square/dense_14/Reluº
$square/Outputs/MatMul/ReadVariableOpReadVariableOp-square_outputs_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$square/Outputs/MatMul/ReadVariableOp¼
square/Outputs/MatMulMatMul"square/dense_14/Relu:activations:0,square/Outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
square/Outputs/MatMul¹
%square/Outputs/BiasAdd/ReadVariableOpReadVariableOp.square_outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%square/Outputs/BiasAdd/ReadVariableOp½
square/Outputs/BiasAddBiasAddsquare/Outputs/MatMul:product:0-square/Outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
square/Outputs/BiasAdd×
IdentityIdentitysquare/Outputs/BiasAdd:output:0&^square/Outputs/BiasAdd/ReadVariableOp%^square/Outputs/MatMul/ReadVariableOp'^square/dense_10/BiasAdd/ReadVariableOp&^square/dense_10/MatMul/ReadVariableOp'^square/dense_11/BiasAdd/ReadVariableOp&^square/dense_11/MatMul/ReadVariableOp'^square/dense_12/BiasAdd/ReadVariableOp&^square/dense_12/MatMul/ReadVariableOp'^square/dense_13/BiasAdd/ReadVariableOp&^square/dense_13/MatMul/ReadVariableOp'^square/dense_14/BiasAdd/ReadVariableOp&^square/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2N
%square/Outputs/BiasAdd/ReadVariableOp%square/Outputs/BiasAdd/ReadVariableOp2L
$square/Outputs/MatMul/ReadVariableOp$square/Outputs/MatMul/ReadVariableOp2P
&square/dense_10/BiasAdd/ReadVariableOp&square/dense_10/BiasAdd/ReadVariableOp2N
%square/dense_10/MatMul/ReadVariableOp%square/dense_10/MatMul/ReadVariableOp2P
&square/dense_11/BiasAdd/ReadVariableOp&square/dense_11/BiasAdd/ReadVariableOp2N
%square/dense_11/MatMul/ReadVariableOp%square/dense_11/MatMul/ReadVariableOp2P
&square/dense_12/BiasAdd/ReadVariableOp&square/dense_12/BiasAdd/ReadVariableOp2N
%square/dense_12/MatMul/ReadVariableOp%square/dense_12/MatMul/ReadVariableOp2P
&square/dense_13/BiasAdd/ReadVariableOp&square/dense_13/BiasAdd/ReadVariableOp2N
%square/dense_13/MatMul/ReadVariableOp%square/dense_13/MatMul/ReadVariableOp2P
&square/dense_14/BiasAdd/ReadVariableOp&square/dense_14/BiasAdd/ReadVariableOp2N
%square/dense_14/MatMul/ReadVariableOp%square/dense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
Ú
}
(__inference_dense_14_layer_call_fn_17534

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
C__inference_dense_14_layer_call_and_return_conditional_losses_167732
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
Ø
|
'__inference_Outputs_layer_call_fn_17553

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
B__inference_Outputs_layer_call_and_return_conditional_losses_167992
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
Ú
}
(__inference_dense_12_layer_call_fn_17470

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
C__inference_dense_12_layer_call_and_return_conditional_losses_167072
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
	

&__inference_square_layer_call_fn_17004

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

unknown_10
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_square_layer_call_and_return_conditional_losses_169772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
	
Û
B__inference_Outputs_layer_call_and_return_conditional_losses_16799

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
	

&__inference_square_layer_call_fn_17374

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

unknown_10
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_square_layer_call_and_return_conditional_losses_170702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

C__inference_dense_14_layer_call_and_return_conditional_losses_17525

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_14/kernel/Regularizer/Square/ReadVariableOp
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
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½

!__inference__traced_restore_17911
file_prefix$
 assignvariableop_dense_10_kernel$
 assignvariableop_1_dense_10_bias&
"assignvariableop_2_dense_11_kernel$
 assignvariableop_3_dense_11_bias&
"assignvariableop_4_dense_12_kernel$
 assignvariableop_5_dense_12_bias&
"assignvariableop_6_dense_13_kernel$
 assignvariableop_7_dense_13_bias&
"assignvariableop_8_dense_14_kernel$
 assignvariableop_9_dense_14_bias&
"assignvariableop_10_outputs_kernel$
 assignvariableop_11_outputs_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1.
*assignvariableop_21_adam_dense_10_kernel_m,
(assignvariableop_22_adam_dense_10_bias_m.
*assignvariableop_23_adam_dense_11_kernel_m,
(assignvariableop_24_adam_dense_11_bias_m.
*assignvariableop_25_adam_dense_12_kernel_m,
(assignvariableop_26_adam_dense_12_bias_m.
*assignvariableop_27_adam_dense_13_kernel_m,
(assignvariableop_28_adam_dense_13_bias_m.
*assignvariableop_29_adam_dense_14_kernel_m,
(assignvariableop_30_adam_dense_14_bias_m-
)assignvariableop_31_adam_outputs_kernel_m+
'assignvariableop_32_adam_outputs_bias_m.
*assignvariableop_33_adam_dense_10_kernel_v,
(assignvariableop_34_adam_dense_10_bias_v.
*assignvariableop_35_adam_dense_11_kernel_v,
(assignvariableop_36_adam_dense_11_bias_v.
*assignvariableop_37_adam_dense_12_kernel_v,
(assignvariableop_38_adam_dense_12_bias_v.
*assignvariableop_39_adam_dense_13_kernel_v,
(assignvariableop_40_adam_dense_13_bias_v.
*assignvariableop_41_adam_dense_14_kernel_v,
(assignvariableop_42_adam_dense_14_bias_v-
)assignvariableop_43_adam_outputs_kernel_v+
'assignvariableop_44_adam_outputs_bias_v
identity_46¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9À
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ì
valueÂB¿.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesê
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_12_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_12_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_14_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_14_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_outputs_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_outputs_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12¥
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14§
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¦
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21²
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_10_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_10_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23²
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_11_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_11_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_12_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_12_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_13_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_13_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_14_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_14_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31±
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_outputs_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¯
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_outputs_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_10_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_10_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_11_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_11_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_12_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_12_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_13_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_13_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41²
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_14_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42°
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_14_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43±
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_outputs_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¯
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_outputs_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¼
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45¯
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*Ë
_input_shapes¹
¶: :::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ê

C__inference_dense_10_layer_call_and_return_conditional_losses_16641

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_10/kernel/Regularizer/Square/ReadVariableOp
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_10_layer_call_fn_17406

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
C__inference_dense_10_layer_call_and_return_conditional_losses_166412
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
\
¥
__inference__traced_save_17766
file_prefix.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop-
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
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop4
0savev2_adam_outputs_kernel_m_read_readvariableop2
.savev2_adam_outputs_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop4
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
ShardedFilenameº
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ì
valueÂB¿.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesä
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesí
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop)savev2_outputs_kernel_read_readvariableop'savev2_outputs_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop0savev2_adam_outputs_kernel_m_read_readvariableop.savev2_adam_outputs_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop0savev2_adam_outputs_kernel_v_read_readvariableop.savev2_adam_outputs_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
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

identity_1Identity_1:output:0*Ë
_input_shapes¹
¶: :@:@:@@:@:@@:@:@@:@:@@:@:@:: : : : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@::@:@:@@:@:@@:@:@@:@:@@:@:@:: 2(
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

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 
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

:@: !

_output_shapes
::$" 

_output_shapes

:@: #

_output_shapes
:@:$$ 

_output_shapes

:@@: %

_output_shapes
:@:$& 

_output_shapes

:@@: '
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

:@: -

_output_shapes
::.

_output_shapes
: 
j


A__inference_square_layer_call_and_return_conditional_losses_17241

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource*
&outputs_matmul_readvariableop_resource+
'outputs_biasadd_readvariableop_resource
identity¢Outputs/BiasAdd/ReadVariableOp¢Outputs/MatMul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢1dense_10/kernel/Regularizer/Square/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢1dense_11/kernel/Regularizer/Square/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢1dense_12/kernel/Regularizer/Square/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢1dense_13/kernel/Regularizer/Square/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢1dense_14/kernel/Regularizer/Square/ReadVariableOp¨
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_10/MatMul/ReadVariableOp
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_10/BiasAdd/ReadVariableOp¥
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/Relu¨
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_11/MatMul/ReadVariableOp£
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOp¥
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_11/BiasAdds
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_11/Relu¨
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp£
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp¥
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/Relu¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/Relu¨
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp¥
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/Relu¥
Outputs/MatMul/ReadVariableOpReadVariableOp&outputs_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
Outputs/MatMul/ReadVariableOp 
Outputs/MatMulMatMuldense_14/Relu:activations:0%Outputs/MatMul/ReadVariableOp:value:0*
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulÎ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulÎ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulÎ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulÎ
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul
IdentityIdentityOutputs/BiasAdd:output:0^Outputs/BiasAdd/ReadVariableOp^Outputs/MatMul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2@
Outputs/BiasAdd/ReadVariableOpOutputs/BiasAdd/ReadVariableOp2>
Outputs/MatMul/ReadVariableOpOutputs/MatMul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Q
£
A__inference_square_layer_call_and_return_conditional_losses_16846

inputs
dense_10_16652
dense_10_16654
dense_11_16685
dense_11_16687
dense_12_16718
dense_12_16720
dense_13_16751
dense_13_16753
dense_14_16784
dense_14_16786
outputs_16810
outputs_16812
identity¢Outputs/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢1dense_10/kernel/Regularizer/Square/ReadVariableOp¢ dense_11/StatefulPartitionedCall¢1dense_11/kernel/Regularizer/Square/ReadVariableOp¢ dense_12/StatefulPartitionedCall¢1dense_12/kernel/Regularizer/Square/ReadVariableOp¢ dense_13/StatefulPartitionedCall¢1dense_13/kernel/Regularizer/Square/ReadVariableOp¢ dense_14/StatefulPartitionedCall¢1dense_14/kernel/Regularizer/Square/ReadVariableOp
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_16652dense_10_16654*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_166412"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_16685dense_11_16687*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_166742"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_16718dense_12_16720*
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
C__inference_dense_12_layer_call_and_return_conditional_losses_167072"
 dense_12/StatefulPartitionedCall´
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_16751dense_13_16753*
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
C__inference_dense_13_layer_call_and_return_conditional_losses_167402"
 dense_13/StatefulPartitionedCall´
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_16784dense_14_16786*
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
C__inference_dense_14_layer_call_and_return_conditional_losses_167732"
 dense_14/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0outputs_16810outputs_16812*
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
B__inference_Outputs_layer_call_and_return_conditional_losses_167992!
Outputs/StatefulPartitionedCallµ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_16652*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulµ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_16685*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulµ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_16718*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulµ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_16751*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulµ
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_14_16784*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulÑ
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall2^dense_11/kernel/Regularizer/Square/ReadVariableOp!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
Q
£
A__inference_square_layer_call_and_return_conditional_losses_16910

inputs
dense_10_16849
dense_10_16851
dense_11_16854
dense_11_16856
dense_12_16859
dense_12_16861
dense_13_16864
dense_13_16866
dense_14_16869
dense_14_16871
outputs_16874
outputs_16876
identity¢Outputs/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢1dense_10/kernel/Regularizer/Square/ReadVariableOp¢ dense_11/StatefulPartitionedCall¢1dense_11/kernel/Regularizer/Square/ReadVariableOp¢ dense_12/StatefulPartitionedCall¢1dense_12/kernel/Regularizer/Square/ReadVariableOp¢ dense_13/StatefulPartitionedCall¢1dense_13/kernel/Regularizer/Square/ReadVariableOp¢ dense_14/StatefulPartitionedCall¢1dense_14/kernel/Regularizer/Square/ReadVariableOp
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_16849dense_10_16851*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_166412"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_16854dense_11_16856*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_166742"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_16859dense_12_16861*
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
C__inference_dense_12_layer_call_and_return_conditional_losses_167072"
 dense_12/StatefulPartitionedCall´
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_16864dense_13_16866*
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
C__inference_dense_13_layer_call_and_return_conditional_losses_167402"
 dense_13/StatefulPartitionedCall´
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_16869dense_14_16871*
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
C__inference_dense_14_layer_call_and_return_conditional_losses_167732"
 dense_14/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0outputs_16874outputs_16876*
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
B__inference_Outputs_layer_call_and_return_conditional_losses_167992!
Outputs/StatefulPartitionedCallµ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_16849*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulµ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_16854*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulµ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_16859*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulµ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_16864*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulµ
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_14_16869*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulÑ
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall2^dense_11/kernel/Regularizer/Square/ReadVariableOp!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ê

C__inference_dense_11_layer_call_and_return_conditional_losses_17429

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_11/kernel/Regularizer/Square/ReadVariableOp
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
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Û
B__inference_Outputs_layer_call_and_return_conditional_losses_17544

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
«
¢
__inference_loss_fn_0_17564>
:dense_10_kernel_regularizer_square_readvariableop_resource
identity¢1dense_10/kernel/Regularizer/Square/ReadVariableOpá
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_10_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul
IdentityIdentity#dense_10/kernel/Regularizer/mul:z:02^dense_10/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp
«
¢
__inference_loss_fn_2_17586>
:dense_12_kernel_regularizer_square_readvariableop_resource
identity¢1dense_12/kernel/Regularizer/Square/ReadVariableOpá
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_12_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul
IdentityIdentity#dense_12/kernel/Regularizer/mul:z:02^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp
Ú
}
(__inference_dense_11_layer_call_fn_17438

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
C__inference_dense_11_layer_call_and_return_conditional_losses_166742
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
õ

#__inference_signature_wrapper_17166

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

unknown_10
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_166202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
Q
£
A__inference_square_layer_call_and_return_conditional_losses_16977

inputs
dense_10_16916
dense_10_16918
dense_11_16921
dense_11_16923
dense_12_16926
dense_12_16928
dense_13_16931
dense_13_16933
dense_14_16936
dense_14_16938
outputs_16941
outputs_16943
identity¢Outputs/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢1dense_10/kernel/Regularizer/Square/ReadVariableOp¢ dense_11/StatefulPartitionedCall¢1dense_11/kernel/Regularizer/Square/ReadVariableOp¢ dense_12/StatefulPartitionedCall¢1dense_12/kernel/Regularizer/Square/ReadVariableOp¢ dense_13/StatefulPartitionedCall¢1dense_13/kernel/Regularizer/Square/ReadVariableOp¢ dense_14/StatefulPartitionedCall¢1dense_14/kernel/Regularizer/Square/ReadVariableOp
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_16916dense_10_16918*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_166412"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_16921dense_11_16923*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_166742"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_16926dense_12_16928*
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
C__inference_dense_12_layer_call_and_return_conditional_losses_167072"
 dense_12/StatefulPartitionedCall´
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_16931dense_13_16933*
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
C__inference_dense_13_layer_call_and_return_conditional_losses_167402"
 dense_13/StatefulPartitionedCall´
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_16936dense_14_16938*
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
C__inference_dense_14_layer_call_and_return_conditional_losses_167732"
 dense_14/StatefulPartitionedCall¯
Outputs/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0outputs_16941outputs_16943*
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
B__inference_Outputs_layer_call_and_return_conditional_losses_167992!
Outputs/StatefulPartitionedCallµ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_16916*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulµ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_16921*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulµ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_16926*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulµ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_16931*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulµ
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_14_16936*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulÑ
IdentityIdentity(Outputs/StatefulPartitionedCall:output:0 ^Outputs/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall2^dense_11/kernel/Regularizer/Square/ReadVariableOp!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2B
Outputs/StatefulPartitionedCallOutputs/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

&__inference_square_layer_call_fn_17097

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

unknown_10
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_square_layer_call_and_return_conditional_losses_170702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ê

C__inference_dense_11_layer_call_and_return_conditional_losses_16674

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_11/kernel/Regularizer/Square/ReadVariableOp
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
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	

&__inference_square_layer_call_fn_17345

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

unknown_10
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_square_layer_call_and_return_conditional_losses_169772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_13_layer_call_fn_17502

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
C__inference_dense_13_layer_call_and_return_conditional_losses_167402
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
C__inference_dense_12_layer_call_and_return_conditional_losses_17461

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_12/kernel/Regularizer/Square/ReadVariableOp
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
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
j


A__inference_square_layer_call_and_return_conditional_losses_17316

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource*
&outputs_matmul_readvariableop_resource+
'outputs_biasadd_readvariableop_resource
identity¢Outputs/BiasAdd/ReadVariableOp¢Outputs/MatMul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢1dense_10/kernel/Regularizer/Square/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢1dense_11/kernel/Regularizer/Square/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢1dense_12/kernel/Regularizer/Square/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢1dense_13/kernel/Regularizer/Square/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢1dense_14/kernel/Regularizer/Square/ReadVariableOp¨
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_10/MatMul/ReadVariableOp
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_10/BiasAdd/ReadVariableOp¥
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/Relu¨
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_11/MatMul/ReadVariableOp£
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOp¥
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_11/BiasAdds
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_11/Relu¨
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp£
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp¥
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/Relu¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/Relu¨
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp¥
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/Relu¥
Outputs/MatMul/ReadVariableOpReadVariableOp&outputs_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
Outputs/MatMul/ReadVariableOp 
Outputs/MatMulMatMuldense_14/Relu:activations:0%Outputs/MatMul/ReadVariableOp:value:0*
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulÎ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOp¶
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_11/kernel/Regularizer/Square
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/Const¾
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/Sum
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_11/kernel/Regularizer/mul/xÀ
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulÎ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp¶
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_12/kernel/Regularizer/Square
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const¾
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_12/kernel/Regularizer/mul/xÀ
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulÎ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulÎ
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul
IdentityIdentityOutputs/BiasAdd:output:0^Outputs/BiasAdd/ReadVariableOp^Outputs/MatMul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2@
Outputs/BiasAdd/ReadVariableOpOutputs/BiasAdd/ReadVariableOp2>
Outputs/MatMul/ReadVariableOpOutputs/MatMul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

C__inference_dense_10_layer_call_and_return_conditional_losses_17397

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_10/kernel/Regularizer/Square/ReadVariableOp
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp¶
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_10/kernel/Regularizer/Square
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const¾
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_10/kernel/Regularizer/mul/xÀ
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

C__inference_dense_13_layer_call_and_return_conditional_losses_16740

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_13/kernel/Regularizer/Square/ReadVariableOp
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
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_13_layer_call_and_return_conditional_losses_17493

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_13/kernel/Regularizer/Square/ReadVariableOp
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
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp¶
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_13/kernel/Regularizer/Square
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const¾
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_13/kernel/Regularizer/mul/xÀ
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

C__inference_dense_14_layer_call_and_return_conditional_losses_16773

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_14/kernel/Regularizer/Square/ReadVariableOp
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
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulË
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«
¢
__inference_loss_fn_4_17608>
:dense_14_kernel_regularizer_square_readvariableop_resource
identity¢1dense_14/kernel/Regularizer/Square/ReadVariableOpá
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_14_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp¶
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_14/kernel/Regularizer/Square
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const¾
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/Sum
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_14/kernel/Regularizer/mul/xÀ
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul
IdentityIdentity#dense_14/kernel/Regularizer/mul:z:02^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp"±L
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:×ò
óC
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
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
}__call__
*~&call_and_return_all_conditional_losses
_default_save_signature"±@
_tf_keras_network@{"class_name": "Functional", "name": "square", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "square", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["Inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dense_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Outputs", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Outputs", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Outputs", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "square", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["Inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dense_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Outputs", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Outputs", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Outputs", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "Inputs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}}
ª

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layeré{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
¬

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¬

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¬

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¬

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ó

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
__call__
+&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "Outputs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Outputs", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
«
2iter

3beta_1

4beta_2
	5decay
6learning_ratememfmgmhmimj mk!ml&mm'mn,mo-mpvqvrvsvtvuvv vw!vx&vy'vz,v{-v|"
	optimizer
v
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11"
trackable_list_wrapper
v
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11"
trackable_list_wrapper
H
0
1
2
3
4"
trackable_list_wrapper
Ê
		variables

trainable_variables
7metrics
regularization_losses
8layer_metrics
9layer_regularization_losses

:layers
;non_trainable_variables
}__call__
_default_save_signature
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
!:@2dense_10/kernel
:@2dense_10/bias
.
0
1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
	variables
<layer_metrics
=metrics
regularization_losses
trainable_variables
>layer_regularization_losses

?layers
@non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_11/kernel
:@2dense_11/bias
.
0
1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
	variables
Alayer_metrics
Bmetrics
regularization_losses
trainable_variables
Clayer_regularization_losses

Dlayers
Enon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_12/kernel
:@2dense_12/bias
.
0
1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
	variables
Flayer_metrics
Gmetrics
regularization_losses
trainable_variables
Hlayer_regularization_losses

Ilayers
Jnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_13/kernel
:@2dense_13/bias
.
 0
!1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
°
"	variables
Klayer_metrics
Lmetrics
#regularization_losses
$trainable_variables
Mlayer_regularization_losses

Nlayers
Onon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_14/kernel
:@2dense_14/bias
.
&0
'1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
°
(	variables
Player_metrics
Qmetrics
)regularization_losses
*trainable_variables
Rlayer_regularization_losses

Slayers
Tnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :@2Outputs/kernel
:2Outputs/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
°
.	variables
Ulayer_metrics
Vmetrics
/regularization_losses
0trainable_variables
Wlayer_regularization_losses

Xlayers
Ynon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
Z0
[1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
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
»
	\total
	]count
^	variables
_	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	`total
	acount
b
_fn_kwargs
c	variables
d	keras_api"Ê
_tf_keras_metric¯{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
\0
]1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
&:$@2Adam/dense_10/kernel/m
 :@2Adam/dense_10/bias/m
&:$@@2Adam/dense_11/kernel/m
 :@2Adam/dense_11/bias/m
&:$@@2Adam/dense_12/kernel/m
 :@2Adam/dense_12/bias/m
&:$@@2Adam/dense_13/kernel/m
 :@2Adam/dense_13/bias/m
&:$@@2Adam/dense_14/kernel/m
 :@2Adam/dense_14/bias/m
%:#@2Adam/Outputs/kernel/m
:2Adam/Outputs/bias/m
&:$@2Adam/dense_10/kernel/v
 :@2Adam/dense_10/bias/v
&:$@@2Adam/dense_11/kernel/v
 :@2Adam/dense_11/bias/v
&:$@@2Adam/dense_12/kernel/v
 :@2Adam/dense_12/bias/v
&:$@@2Adam/dense_13/kernel/v
 :@2Adam/dense_13/bias/v
&:$@@2Adam/dense_14/kernel/v
 :@2Adam/dense_14/bias/v
%:#@2Adam/Outputs/kernel/v
:2Adam/Outputs/bias/v
æ2ã
&__inference_square_layer_call_fn_17345
&__inference_square_layer_call_fn_17004
&__inference_square_layer_call_fn_17097
&__inference_square_layer_call_fn_17374À
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
Ò2Ï
A__inference_square_layer_call_and_return_conditional_losses_17316
A__inference_square_layer_call_and_return_conditional_losses_17241
A__inference_square_layer_call_and_return_conditional_losses_16846
A__inference_square_layer_call_and_return_conditional_losses_16910À
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
 __inference__wrapped_model_16620µ
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
Ò2Ï
(__inference_dense_10_layer_call_fn_17406¢
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
C__inference_dense_10_layer_call_and_return_conditional_losses_17397¢
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
(__inference_dense_11_layer_call_fn_17438¢
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
C__inference_dense_11_layer_call_and_return_conditional_losses_17429¢
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
(__inference_dense_12_layer_call_fn_17470¢
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
C__inference_dense_12_layer_call_and_return_conditional_losses_17461¢
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
(__inference_dense_13_layer_call_fn_17502¢
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
C__inference_dense_13_layer_call_and_return_conditional_losses_17493¢
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
(__inference_dense_14_layer_call_fn_17534¢
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
C__inference_dense_14_layer_call_and_return_conditional_losses_17525¢
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
'__inference_Outputs_layer_call_fn_17553¢
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
B__inference_Outputs_layer_call_and_return_conditional_losses_17544¢
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
__inference_loss_fn_0_17564
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
__inference_loss_fn_1_17575
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
__inference_loss_fn_2_17586
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
__inference_loss_fn_3_17597
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
__inference_loss_fn_4_17608
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
#__inference_signature_wrapper_17166Inputs"
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
B__inference_Outputs_layer_call_and_return_conditional_losses_17544\,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_Outputs_layer_call_fn_17553O,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
 __inference__wrapped_model_16620r !&',-/¢,
%¢"
 
Inputsÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
Outputs!
Outputsÿÿÿÿÿÿÿÿÿ£
C__inference_dense_10_layer_call_and_return_conditional_losses_17397\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_10_layer_call_fn_17406O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_11_layer_call_and_return_conditional_losses_17429\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_11_layer_call_fn_17438O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_12_layer_call_and_return_conditional_losses_17461\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_12_layer_call_fn_17470O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_13_layer_call_and_return_conditional_losses_17493\ !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_13_layer_call_fn_17502O !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_14_layer_call_and_return_conditional_losses_17525\&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_14_layer_call_fn_17534O&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@:
__inference_loss_fn_0_17564¢

¢ 
ª " :
__inference_loss_fn_1_17575¢

¢ 
ª " :
__inference_loss_fn_2_17586¢

¢ 
ª " :
__inference_loss_fn_3_17597 ¢

¢ 
ª " :
__inference_loss_fn_4_17608&¢

¢ 
ª " £
#__inference_signature_wrapper_17166| !&',-9¢6
¢ 
/ª,
*
Inputs 
Inputsÿÿÿÿÿÿÿÿÿ"1ª.
,
Outputs!
Outputsÿÿÿÿÿÿÿÿÿ³
A__inference_square_layer_call_and_return_conditional_losses_16846n !&',-7¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
A__inference_square_layer_call_and_return_conditional_losses_16910n !&',-7¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
A__inference_square_layer_call_and_return_conditional_losses_17241n !&',-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
A__inference_square_layer_call_and_return_conditional_losses_17316n !&',-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_square_layer_call_fn_17004a !&',-7¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_square_layer_call_fn_17097a !&',-7¢4
-¢*
 
Inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_square_layer_call_fn_17345a !&',-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_square_layer_call_fn_17374a !&',-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ