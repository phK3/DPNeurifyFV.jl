
# TODO: probably better to add a new field to each node!!!

linear_nodes = Dict(DummyInputNode => false, # it is really linear, but we wouldn't want to convert it
                    Linear => true,
                    AddConst => true,
                    SubConst => true,
                    Concat => true,  # should we really convert that? Maybe have a special rule to only change dims to concat along?
                    Convolution => true,
                    ConvolutionTranspose => true,
                    AveragePool => true,
                    DropoutLayer => true,  # maybe just do nothing here?
                    Reshape => true,
                    Flatten => true,
                    BatchNormalization => true,
                    Upsampling => false,  # only true when nearest mode is used! Need special method
                    Add => true,
                    Sub => true,
                    Gather => true,
                    Slice => true,
                    SplitNode => true,
                    Transpose => true,
                    Squeeze => true,
                    LSTMCell => false,
                    LSTMLayer => false,
                    Relu => false,
                    Sigmoid => false,
                    Tanh => false,
                    Softmax => false,
                    )

# TODO: check, if this is correct!
batched_nodes = Dict(DummyInputNode => true,
                    Linear => true,
                    AddConst => true,
                    SubConst => true,
                    Concat => false,  # can't concat tensors, when one has batch dim and the others don't
                    Convolution => true,
                    ConvolutionTranspose => true,
                    AveragePool => true,
                    DropoutLayer => true,  
                    Reshape => true, 
                    Flatten => true,
                    BatchNormalization => true,
                    Upsampling => true, 
                    Add => true,
                    Sub => true,
                    Gather => false,  # TODO: modify my_gather to handle batches
                    Slice => true,
                    SplitNode => true,
                    Transpose => true,
                    Squeeze => true,
                    LSTMCell => true,
                    LSTMLayer => true,
                    Relu => true,
                    Sigmoid => true,
                    Tanh => true,
                    Softmax => true,
                    )