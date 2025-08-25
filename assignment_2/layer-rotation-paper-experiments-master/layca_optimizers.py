'''
Code for applying Layca on SGD, Adam, RMSprop and Adagrad.
Updated for Keras 3 compatibility.
'''

import keras
from keras.optimizers import Optimizer
import keras.ops as ops
import tensorflow as tf
import numpy as np

def layca(p, step, lr):
    '''
    Core operations of layca.
    Takes the current parameters and the step computed by an optimizer, and 
         - projects and normalizes the step such that the rotation operated on the layer's weights is controlled
         - after the step has been taken, recovers initial norms of the parameters
    '''
    if 'kernel' in p.name: # only kernels are optimized when using Layca (and not biases and batchnorm parameters)
        # projecting step on tangent space of sphere -> orthogonal to the parameters p
        initial_norm = ops.norm(p)
        step = step - (ops.sum(step * p))* p / initial_norm**2

        # normalizing step size
        step = ops.cond(ops.norm(step)<= keras.backend.epsilon(), lambda: ops.zeros_like(step), lambda: step/ (ops.norm(step)) * initial_norm)
        
        # applying step
        new_p =  p - lr * step

        # recovering norm of the parameter from before the update
        new_p = new_p / ops.norm(new_p) * initial_norm
        return new_p
    else:
        return p 
            
class SGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        nesterov: boolean. Whether to apply Nesterov momentum.
        multipliers: dictionary with as keys layer names and values the corresponding layer-wise learning rate multiplier
        adam_like_momentum: boolean, if a momentum scheme similar to adam should be used
        layca: boolean, wether to apply layca or not
    """

    def __init__(self, learning_rate=0.01, momentum=0., 
                 nesterov=False, multipliers={'$ùµµ':1.}, adam_like_momentum = False, 
                 layca = False, normalized = False, effective_lr = False, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.adam_like_momentum = adam_like_momentum
        self.multipliers = multipliers
        self.layca = layca
        self.normalized = normalized
        self.effective_lr = effective_lr

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        """One step of SGD."""
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        
        # Find the corresponding momentum variable
        m = None
        for i, var in enumerate(self.variables):
            if var.name == variable.name:
                if i < len(self.momentums):
                    m = self.momentums[i]
                break
        
        if m is None:
            # Create momentum variable on the fly if not found
            m = ops.zeros_like(variable)
        
        # Apply layer-wise learning rate multipliers
        processed = False
        for key in self.multipliers.keys():
            if key+'/' in variable.name and not processed:
                learning_rate = learning_rate * self.multipliers[key]
                processed = True
        
        if self.adam_like_momentum:
            v = (self.momentum * m) - (1. - self.momentum) * gradient
            if m is not None and hasattr(m, 'assign'):
                m.assign(v)
            v = learning_rate * v
        else:
            v = self.momentum * m - learning_rate * gradient
            if m is not None and hasattr(m, 'assign'):
                m.assign(v)
         
        if self.nesterov:
            step = self.momentum * v - learning_rate * gradient
        else:
            step = v
        
        if self.layca:
            new_p = layca(variable, -step, learning_rate)
        elif self.normalized:
            step = ops.cond(ops.norm(step)<= keras.backend.epsilon(), 
                          lambda: ops.zeros_like(step), 
                          lambda: step/ (ops.norm(step)))
            new_p = variable + learning_rate * step
        elif self.effective_lr:
            new_p = variable + step * ops.norm(variable)**2
        else:
            new_p = variable + step

        self.assign(variable, new_p)

    def get_config(self):
        config = super().get_config()
        config.update({
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "adam_like_momentum": self.adam_like_momentum,
            "multipliers": self.multipliers,
            "layca": self.layca,
            "normalized": self.normalized,
            "effective_lr": self.effective_lr,
        })
        return config

class RMSprop(Optimizer):
    """RMSProp optimizer."""

    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7, layca=False, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.rho = rho
        self.epsilon = epsilon
        self.layca = layca

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.accumulators = []
        for var in var_list:
            self.accumulators.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="accumulator"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        """One step of RMSprop."""
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        
        # Find the corresponding accumulator
        a = None
        for i, var in enumerate(self.variables):
            if var.name == variable.name:
                a = self.accumulators[i]
                break
        
        if a is None:
            a = ops.zeros_like(variable)

        # Update accumulator
        new_a = self.rho * a + (1. - self.rho) * ops.square(gradient)
        self.assign(a, new_a)
        
        step = learning_rate * gradient / (ops.sqrt(new_a) + self.epsilon)
        
        if self.layca:
            new_p = layca(variable, step, learning_rate)
        else:
            new_p = variable - step
        
        self.assign(variable, new_p)

    def get_config(self):
        config = super().get_config()
        config.update({
            "rho": self.rho,
            "epsilon": self.epsilon,
            "layca": self.layca,
        })
        return config

class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, 
                 epsilon=1e-7, layca=False, amsgrad=False, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.layca = layca
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.ms = []
        self.vs = []
        if self.amsgrad:
            self.vhats = []
        for var in var_list:
            self.ms.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="m"
                )
            )
            self.vs.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="v"
                )
            )
            if self.amsgrad:
                self.vhats.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="vhat"
                    )
                )

    def update_step(self, gradient, variable, learning_rate):
        """One step of Adam."""
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        
        # Find corresponding variables
        m = v = vhat = None
        for i, var in enumerate(self.variables):
            if var.name == variable.name:
                m = self.ms[i]
                v = self.vs[i]
                if self.amsgrad:
                    vhat = self.vhats[i]
                break

        if m is None or v is None:
            m = ops.zeros_like(variable)
            v = ops.zeros_like(variable)

        # Bias correction
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(self.beta_1, local_step)
        beta_2_power = ops.power(self.beta_2, local_step)
        
        lr_t = learning_rate * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Update moments
        m_t = self.beta_1 * m + (1. - self.beta_1) * gradient
        v_t = self.beta_2 * v + (1. - self.beta_2) * ops.square(gradient)
        
        self.assign(m, m_t)
        self.assign(v, v_t)

        if self.amsgrad:
            vhat_t = ops.maximum(vhat, v_t)
            self.assign(vhat, vhat_t)
            step = lr_t * m_t / (ops.sqrt(vhat_t) + self.epsilon)
        else:
            step = lr_t * m_t / (ops.sqrt(v_t) + self.epsilon)

        if self.layca:
            new_p = layca(variable, step, learning_rate)
        else:
            new_p = variable - step

        self.assign(variable, new_p)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
            "layca": self.layca,
        })
        return config

class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(self, learning_rate=0.01, epsilon=1e-7, layca=False, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.epsilon = epsilon
        self.layca = layca

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.accumulators = []
        for var in var_list:
            self.accumulators.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="accumulator"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        """One step of Adagrad."""
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        
        # Find corresponding accumulator
        a = None
        for i, var in enumerate(self.variables):
            if var.name == variable.name:
                a = self.accumulators[i]
                break

        if a is None:
            a = ops.zeros_like(variable)

        # Update accumulator
        new_a = a + ops.square(gradient)
        self.assign(a, new_a)
        
        step = learning_rate * gradient / (ops.sqrt(new_a) + self.epsilon)
        
        if self.layca:
            new_p = layca(variable, step, learning_rate)
        else:
            new_p = variable - step
        
        self.assign(variable, new_p)

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "layca": self.layca,
        })
        return config