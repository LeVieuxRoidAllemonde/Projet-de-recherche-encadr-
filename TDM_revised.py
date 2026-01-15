import numpy as np
from scipy.fftpack import shift
import torch
from torch.nn import Module, Parameter
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric
from torch.distributions.binomial import Binomial
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.gamma import Gamma


# in order to print the whole arrays
def fullnp():
    return np.printoptions(threshold=np.inf)

# small epsilon for logs
_eps = 1e-50


def binom_logprob_manual_kNk(k, N, p):
    """
    Computes the log-probability of a Binomial(N, p) distribution for possibly
    non-integer and differentiable arguments, used when PyTorch's built-in 
    Binomial distribution cannot handle fractional or tensor-valued total counts.

    This function provides a numerically stable manual implementation of the
    Binomial log-probability:

        log P(k | N, p) = log( C(N, k) * p^k * (1 - p)^(N - k) )

    where C(N, k) = Γ(N + 1) / (Γ(k + 1) * Γ(N - k + 1)) is computed via the
    log-gamma function to support continuous relaxation of N and k.

    In the Tonal Diffusion Model, this function is used to define smooth
    Binomial-like path-length probabilities even when the number of steps (N)
    is fractional due to learned parameters, avoiding domain errors in the
    standard discrete Binomial implementation.
    """
    p_clamped = torch.clamp(p, _eps, 1.0 - _eps)
    log_choose = torch.lgamma(N + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(N - k + 1.0)
    return log_choose + k * torch.log(p_clamped) + (N - k) * torch.log(1.0 - p_clamped)

class IntervalClassModel(Module): 
    @staticmethod
    def dkl(p, q, dim=None):
        """
        Robust D_{KL}(p || q). If q has a longer support in dimension 1 than p,
        compute KL(p || q_window) for every contiguous window of q of width p.shape[1]
        and return the minimum KL per data point. This avoids shape-mismatch crashes
        when q is the full-support distribution and p is the observed window.
        """
        # ensure same dtype
        p = p.to(dtype=torch.float64)
        q = q.to(dtype=torch.float64)

        # quick path: same shape => normal KL
        if p.shape == q.shape:
            zeros = torch.zeros_like(p)
            dkl = torch.where(torch.isclose(p, zeros), zeros, p * (p.log() - q.log())).sum(dim=dim)
            if np.any(torch.isnan(dkl).data.numpy()):
                print("dkl debug p.shape", p.shape, "q.shape", q.shape)
                print(dkl)
                raise RuntimeWarning("Got nan")
            return dkl

        # otherwise expect q.shape[0] == p.shape[0] (batch dim equal) and
        # q.shape[1] >= p.shape[1] (q has larger support)
        if q.dim() < 2 or p.dim() < 2:
            raise RuntimeError(f"dkl: unexpected tensor dims p.dim={p.dim()}, q.dim={q.dim()}")

        if q.shape[0] != p.shape[0]:
            # try to broadcast batch if needed
            if q.shape[0] == 1 and p.shape[0] > 1:
                q = q.expand(p.shape[0], *q.shape[1:])
            else:
                raise RuntimeError(f"dkl: batch size mismatch p.shape={p.shape}, q.shape={q.shape}")

        n = p.shape[1]
        m = q.shape[1]
        if m < n:
            raise RuntimeError(f"dkl: q shorter than p (q:{m} < p:{n}) -- cannot match")

        # unfold q along dim=1 into sliding windows of width n
        # q_unf shape: (batch, n_windows, n, ...)
        q_unf = q.unfold(dimension=1, size=n, step=1)  # (B, W, n, ...)
        # normalize each window along the 'n' axis
        # sum over axis=2 (the window axis)
        sum_q_unf = q_unf.sum(dim=2, keepdim=True)
        # prevent division by zero
        sum_q_unf = sum_q_unf + 1e-50
        q_unf = q_unf / sum_q_unf

        # prepare p expanded to (B, W, n, ...)
        p_exp = p.unsqueeze(1)  # (B, 1, n, ...)
        zeros = torch.zeros_like(p_exp)

        # compute elementwise KL term; broadcasting will align dimensions
        # take log of q_unf safely (q_unf > 0 because of normalization and tiny epsilon)
        term = torch.where(torch.isclose(p_exp, zeros),
                        zeros,
                        p_exp * (p_exp.log() - q_unf.log()))
        # sum over the 'n' axis -> (B, W, ...)
        dkl_windows = term.sum(dim=2)

        # if there are extra trailing dims after axis 2, we sum them as well
        # (this handles potential latent dims; reduce them to per-window scalars)
        if dkl_windows.dim() > 2:
            # sum over all remaining dims except batch and window dims
            reduce_dims = tuple(range(2, dkl_windows.dim()))
            dkl_windows = dkl_windows.sum(dim=reduce_dims)

        # pick minimal window (best alignment) per batch element
        dkl_min, _ = dkl_windows.min(dim=1)  # shape (B,)
        if not torch.isfinite(dkl_min).all():
            print("dkl: non-finite values", dkl_min)
            raise RuntimeWarning("Got non-finite (inf/nan)")
        return dkl_min


    @staticmethod
    def dkl_log(log_p, log_q, dim=None):
        """
        Same as above but accepts log-space inputs. If shapes mismatch, operate
        on sliding windows of q in exp(log_q) space.
        """
        # quick path: same shape
        if log_p.shape == log_q.shape:
            zeros = torch.zeros_like(log_p)
            dkl = torch.where(torch.isfinite(log_p),
                            log_p.exp() * (log_p - log_q),
                            zeros).sum(dim=dim)
            if not np.all(torch.isfinite(dkl.data.numpy())):
                print("dkl_log debug", dkl)
                raise RuntimeWarning("Got non-finite (inf/nan)")
            return dkl

        # otherwise do windowed version similarly to dkl
        p = log_p.exp()
        q = log_q.exp()
        # reuse dkl implementation (it already handles normalization and windowing)
        return IntervalClassModel.dkl(p, q, dim=dim)
    

    def __init__(self): 
        super().__init__()
        self.iteration = 0 # initialisation

    def forward(self, *input):
        raise RuntimeWarning("This is not a normal Module") 

    def set_data(self, data, weights=None):
        self.iteration = 0
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)
        self.n_data, self.n_interval_classes = data.shape
        # set data weights if specified (to weigth composers equally when calculating the mean loss on a whole corpus)
        if weights is not None:
            self.data_weights = torch.from_numpy(weights)
            self.data_weights = self.data_weights / self.data_weights.sum()
        else:
            self.data_weights = None
        

    def get_params(self, **kwargs):
        return np.concatenate(tuple(p.data.numpy().flatten() for p in self.parameters(**kwargs)))
        #  Allows to retrieve all model parameters (in the form of a numpy vector).
    def set_params(self, params, **kwargs):
        if np.any(np.isnan(params)):
            raise RuntimeWarning(f"nan params: {params}") # check that there is no nan in the parameters
        if len(params.shape) > 1:
            # expect all shapes to match
            for p, n in zip(self.parameters(**kwargs), params):
                p.data = torch.from_numpy(n)
        else:
            # reshape consecutive segments
            idx = 0
            for p in self.parameters(**kwargs):
                size = np.prod(p.shape)
                p.data = torch.from_numpy(params[idx:idx + size].reshape(p.shape))
                idx += size

    def _loss(self):
        raise NotImplementedError  # abstract method each model will have its own loss

    def loss(self, params, **kwargs): 
        self.set_params(params, **kwargs)
        self.zero_grad()
        with torch.no_grad():
            return self._loss().data.numpy() 

    def grad(self, params, *, return_loss=False, **kwargs): # retropropagation
        self.set_params(params, **kwargs)
        self.zero_grad()
        loss = self._loss()
        loss.backward()
        grad = np.concatenate(tuple(p.grad.data.numpy().flatten() for p in self.parameters(**kwargs)))
        if np.any(np.isnan(grad)):
            with fullnp():
                raise RuntimeWarning(f"grad has nan values: {grad}\nloss: {loss}")
        if return_loss:
            return loss.data.numpy().copy(), grad
        else:
            return grad

    def closure(self):
        self.zero_grad()
        loss = self._loss()
        loss.backward()
        return loss

    def callback(self, params):
        self.iteration += 1
        print(f"iteration {self.iteration}")
        print(f"    loss: {self.loss(params)}")
        # to print the loss at each iteration

class TonnetzModel(IntervalClassModel):

    def __init__(self,
                 interval_steps=(
                         1,   # fifth up
                         -1,  # fifth down
                         -3,  # minor third up
                         3,   # minor third down
                         4,   # major third up
                         -4   # major third down
                         #6,  tritone up
                         #-6 # tritone down
                 ),
                 margin=0.5, #Used to build an ‘extended support’ around interval classes to allow for shifts 
                 latent_shape=(), # Allows additional latent variables to be added (e.g. multiple tonal origins).
                 soft_max_posterior=False,
                 separate_parameters=False,
                 *args,
                 **kwargs):
        super().__init__()
        self.interval_steps = np.array(interval_steps)
        self.n_interval_steps = len(interval_steps)
        self.margin = margin
        self.n_data = None
        self.n_interval_classes = None
        self.n_dist_support = None
        self.latent_shape = latent_shape
        self.soft_max_posterior = soft_max_posterior
        self.separate_parameters = separate_parameters
        self.reference_center = None
        self.n_shifts = None
        self.interval_class_distribution = None
        self.data = None
        self.matched_dist = None
        self.matched_loss = None
        self.matched_shift = None
        self.matched_latent = None
        self.data_weights = None
        self.neg_log_likes = None
        if self.soft_max_posterior:
            self.beta = Parameter(torch.tensor([]))
            # self.neg_log_likes.shape = (n_data, n_shifts, ...)  pieces and transpositions
    def _loss(self):
        self.match_distributions() 
        # In the Bayesian case, we do not make a decision for a particular transport; we look at the posterior distribution of all cases. There is no priority; it is just a normalised softmax, taking into account the scores obtained.
        if self.soft_max_posterior: # idea of considering all different transpositions as posterior distributions (a softmax distribution)
            # compute posterior of latent variables
            latent_dims = tuple(range(1, len(self.neg_log_likes.shape)))  # including transposition/shift!
            latent_log_posterior = -self.neg_log_likes * self.beta.exp() # beta c'est un poid c'est la température si grand la distribution softmax avantage la meilleure transposition à chaque fois meileur choix local ; si petit moyenne pour choisir une transposition
            latent_log_posterior = latent_log_posterior - latent_log_posterior.logsumexp(dim=latent_dims, keepdim=True) 
            # compute marginal neg-log-likelihood (per piece)
            # the weighted average per piece, with the weights being the calculated probabilities 
            neg_log_like = -(-self.neg_log_likes + latent_log_posterior).logsumexp(dim=latent_dims) # pour chaque shift calcule une probabilité d'être obetnu de façon conjointe avec tels latents
        else:
            data_slice = np.array(list(range(self.n_data))) # pieces indexation
            if self.matched_latent is None: # if no latents 
                neg_log_like = self.neg_log_likes[data_slice, self.matched_shift] # pour chaque pièce, on prend la valeur neg_log_likes correspondant au shift choisi (le meilleur alignement trouvé).
            else: # So for each piece, we obtain a cost that reflects the entire distribution of shifts/latencies, not just the best one.
                neg_log_like = self.neg_log_likes[(data_slice, self.matched_shift) + tuple(self.matched_latent)] # if latent, add the indices of the corresponding latents 
        if self.data_weights is None:
            return neg_log_like.mean() # one can choose to use a weighted sum or simple average for the final loss
        else:
            return (neg_log_like * self.data_weights).sum()

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # get necessary support of distribution and reference center
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            self.reference_center = None
            self.n_dist_support = self.n_interval_classes
            self.n_shifts = self.n_interval_classes
        else:
            self.n_dist_support = int(np.ceil((2 * self.margin + 1) * self.n_interval_classes))
            self.reference_center = int(np.round((self.margin + 0.5) * self.n_interval_classes))
            self.n_shifts = self.n_dist_support - self.n_interval_classes + 1
        if self.soft_max_posterior:
            self.beta.data = torch.tensor([0.])

    def get_interpretable_params(self, *args, **kwargs):
        d = dict(
            # loss=self.matched_loss.data.numpy().copy(),
            shift=self.matched_shift
        )
        if self.soft_max_posterior:
            d = dict(**d, beta=[self.beta.exp().data.numpy()[0] for _ in range(self.n_data)])
        if self.matched_latent is not None:
            d = dict(**d, **{f"latent_{idx+1}": latent for idx, latent in enumerate(self.matched_latent)})
        return d


    def match_distributions(self): # shifts is used to transpose the distribution of the tonnetz into the 12 notes of the chromatic scale
        #print("DEBUG 0: entering match_distributions")
        if hasattr(self, "n_shifts"):
            n_shifts = self.n_shifts
        else:
            n_shifts = self.n_dist_support - self.n_interval_classes + 1 # number of shifts tested so far
        #print("DEBUG 1: got n_shifts =", n_shifts) 
        all_data_indices = np.arange(self.n_data)
        #print("DEBUG 2: all_data_indices shape =", all_data_indices.shape)
        if hasattr(self, "separate_parameters") and self.separate_parameters: 
            self.neg_log_likes = self.dkl(self.data[:, :, None], self.interval_class_distribution[:, :, :], dim=1) # calculates the divergence between the model distribution and the data distribution
            self.matched_shift = np.argmin(self.neg_log_likes.data.numpy(), axis=1) # indice of best distribution
            self.matched_dist = self.interval_class_distribution[all_data_indices, :, self.matched_shift] # corresponding distribution
            self.matched_loss = self.neg_log_likes[all_data_indices, self.matched_shift] # corresponding loss
        else: # cases where a part has not yet been processed by the model (initialisations)
            self.neg_log_likes = torch.zeros((self.n_data, n_shifts) + self.latent_shape, dtype=torch.float64)
            latent_none = tuple(None for _ in self.latent_shape)
            latent_slice = tuple(slice(None) for _ in self.latent_shape)
            # adding a hack if weird shape
            if len(self.interval_class_distribution.shape) == 3 and \
                self.interval_class_distribution.shape[1] == self.n_data:
                #print("Fixing interval_class_distribution shape automatically")
                self.interval_class_distribution = self.interval_class_distribution[:, 0, :]
            #print("interval_class_distribution shape:", self.interval_class_distribution.shape)
            #print("expected n_dist_support:", self.n_dist_support)
            #print("expected n_interval_classes:", self.n_interval_classes)
            #print("n_shifts:", n_shifts)
            for shift in range(n_shifts):
                # keep profile dimension
                dist = self.interval_class_distribution[(slice(None),
                                                         slice(shift, shift + self.n_interval_classes)) +
                                                        latent_slice] #  distribution for a specific shift
                dist = dist / dist.sum(dim=1, keepdim=True) # normalise
                # neg-log-likelihoods for all profiles (and all data as in parent function)
                nll = self.dkl(self.data[(slice(None), slice(None)) + latent_none], dist, dim=1) # KL divergence for a specific shift
                self.neg_log_likes[(slice(None), shift) + latent_slice] = nll
                if self.latent_shape:
                    matched_latent = np.unravel_index(
                        np.argmin(nll.data.numpy().reshape(self.n_data, -1), axis=1),
                        self.latent_shape
                    ) # choice of the best shift
                    matched_dist = dist[(0, slice(None)) + matched_latent].transpose(0, 1) # linked distribution
                else:
                    matched_latent = ()
                    matched_dist = dist
                matched_data_indices = (all_data_indices,) + matched_latent
                if shift == 0:
                    self.matched_dist = matched_dist
                    self.matched_loss = nll[matched_data_indices]
                    self.matched_shift = np.zeros(self.n_data, dtype=int)
                    if self.latent_shape:
                        self.matched_latent = matched_latent # cas où ça change rien
                else:
                    cond = nll[matched_data_indices] < self.matched_loss

                    try:
                        #print("shift:", shift)
                        #print("cond shape:", cond.shape)
                        #print("matched_dist shape:", matched_dist.shape)
                        #print("self.matched_dist shape:", self.matched_dist.shape)

                        self.matched_dist = torch.where(cond[:, None], matched_dist, self.matched_dist)
                        self.matched_loss = torch.where(cond, nll[matched_data_indices], self.matched_loss)
                        self.matched_shift = np.where(cond, shift, self.matched_shift)
                        if self.latent_shape:
                            self.matched_latent = np.where(cond, matched_latent, self.matched_latent)

                    except Exception as e:
                        # debug
                        print("ERROR during matching distributions")
                        print("shift:", shift)
                        print("cond shape:", getattr(cond, 'shape', None))
                        print("matched_dist shape:", getattr(matched_dist, 'shape', None))
                        print("self.matched_dist shape:", getattr(self.matched_dist, 'shape', None))
                        print("Error type:", type(e).__name__)
                        print("Error message:", e)
                        raise


    def get_results(self, *args, **kwargs):
        """
        Compute model loss and return internal results
        :return: tuple of arrays (matched distribution, loss, center) containing results per data point
        """
        # compute everything
        self._loss()
        # return internal results
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            return (self.matched_dist.data.numpy().copy(),
                    self.matched_loss.data.numpy().copy(),
                    self.matched_shift)
        else:
            return (self.matched_dist.data.numpy().copy(),
                    self.matched_loss.data.numpy().copy(),
                    self.reference_center - self.matched_shift)


class DiffusionModel(TonnetzModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_interval_class_distribution = None # initial distribution (original tonal centre)
        self.transition_matrix = None # transition matrix on the Tonnetz

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise interval class distributions with single tonal center
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            self.init_interval_class_distribution = np.zeros((self.n_data,
                                                              self.n_interval_classes,
                                                              self.n_interval_classes))
            # For each piece and each interval class, an identity matrix is initialised.
            self.init_interval_class_distribution[:,
            np.arange(self.n_interval_classes),
            np.arange(self.n_interval_classes)] = 1
        else:
             # this p(to|c): we initialise the entire probability mass on the tonal centre for the starting weight of each path
            self.init_interval_class_distribution = np.zeros((self.n_data, self.n_dist_support))
            self.init_interval_class_distribution[:, self.reference_center] = 1
        self.init_interval_class_distribution = torch.from_numpy(self.init_interval_class_distribution)
        # init transition matrix
        self.transition_matrix = np.zeros((self.n_dist_support, self.n_dist_support, self.n_interval_steps))
        for interval_index, interval_step in enumerate(self.interval_steps):
            if interval_step > 0: # ascending steps
                from_indices = np.arange(0, self.n_dist_support - interval_step) # This is understandable given that dist_support is the total number of note classes.
            else: # descending steps
                from_indices = np.arange(-interval_step, self.n_dist_support) 
            to_indices = from_indices + interval_step # we go from the starting note index to the ending note by traversing a given interval
            self.transition_matrix[from_indices, to_indices, interval_index] = 1 # encoded by 1 in the matrix
        self.transition_matrix = torch.from_numpy(self.transition_matrix)

    def perform_diffusion(self):
        raise NotImplementedError

    def _loss(self):
        # diffusion before calculating loss !
        self.perform_diffusion()
        return super()._loss() 


class TonalDiffusionModel(DiffusionModel):

    def __init__(self,
                 min_iterations=None,
                 max_iterations=1000, # In theory, we sum over any n, but here the maximum threshold is still 1000.
                 path_dist=Binomial, #binomial distribution of the number of steps on the tone network (default)
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.effective_min_iterations = None
        self.log_interval_step_weights = Parameter(torch.tensor([])) 
        self.path_log_params = Parameter(torch.tensor([])) # if binomial law, we learn a logit of p 
        self.path_dist = path_dist
        if path_dist == Gamma:
            self.precompute_path_dist = True # if gamma, we precompute because discretisation of the continuous values of the gamma function is complicated (integrate the intervals [ n ; n+1 [) 
        else:
            self.precompute_path_dist = False

    def set_data(self, *args, **kwargs): 
        super().set_data(*args, **kwargs)
        # set minimum number of iterations to reach every point
        if self.min_iterations is None:
            largest_step_down = -min(np.min(self.interval_steps), 0)
            largest_step_up = max(np.max(self.interval_steps), 0) 
            if hasattr(self, "separate_parameters") and self.separate_parameters:
                self.effective_min_iterations = int(np.ceil(
                    self.n_interval_classes / min(largest_step_up, largest_step_down)
                )) + 1 # calculation of the minimum number of iterations to explore the entire tone network division of the number because if we take steps of a defined interval, how many iterations are needed at a minimum on this interval to reach the furthest note, roughly speaking
            else:
                self.effective_min_iterations = int(np.ceil(max(
                    self.reference_center / largest_step_down,
                    (self.n_dist_support - self.reference_center) / largest_step_up
                ))) + 1
        else:
            self.effective_min_iterations = self.min_iterations
        # initialise weights
        if self.n_interval_steps == 6:
            weights = np.array([3, 3, 0, 0, 0, 0], dtype=np.float64) # initially, more emphasis is placed on fifths
        else:
            weights = np.zeros(self.n_interval_steps, dtype=np.float64) # if we choose more intervals than in the classical model here, they initialise everything to zero
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            full_weights = np.zeros((self.n_data, self.n_interval_steps, self.n_shifts), dtype=np.float64)
            full_weights[:, :, :] = weights[None, :, None] # Fill in the parameter tensor for all parts and transpositions
        else:
            full_weights = np.zeros((self.n_data, self.n_interval_steps), dtype=np.float64)
            full_weights[:, :] = weights[None, :] 
        self.log_interval_step_weights.data = torch.from_numpy(full_weights)
        # initialise distribution parameters 
        # default values
        if self.path_dist in [Poisson, Geometric]:
            default_params = [1]
            # np.torch.zeros(self.n_data, dtype=torch.float64)
        elif self.path_dist in [Gamma, Binomial, NegativeBinomial]:
             # default_params = np.ones(2, dtype=np.float64)
             if self.path_dist == Gamma:
                 default_params = [2, -2]
                 # default_params[:, 0] *= 2
                 # default_params[:, 1] *= -2
             elif self.path_dist == Binomial:
                 default_params = [2, 0]
                 # default_params[:, 0] *= 2
                 # default_params[:, 1] *= 0
             else:
                 default_params = [0, 0]
                 # default_params *= 0
             # self.path_log_params.data = torch.from_numpy(default_params)
        else:
            raise RuntimeWarning("Failed Case")
        default_params = np.array(default_params, dtype=np.float64) 
        # fill # a nouveau on fill toutes les matrices de paramètres pas des poids mais des distributions ici de la longueur du chemin
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            full_params = np.zeros((self.n_data, len(default_params), self.n_shifts), dtype=np.float64)
            full_params[...] = default_params[None, :, None] 
        else:
            full_params = np.zeros((self.n_data, len(default_params)), dtype=np.float64) 
            full_params[...] = default_params[None, :]
        self.path_log_params.data = torch.from_numpy(full_params)

    def get_results(self, shifts=None, *args, **kwargs):
        ret = super().get_results(*args, **kwargs)
        if shifts is None:
            return ret # retourne the distribution for the best shift
        else:
            return (self.interval_class_distribution[np.arange(self.n_data), :, shifts].data.numpy().copy(),
                    self.neg_log_likes[np.arange(self.n_data), shifts].data.numpy().copy(),
                    shifts) # returns the distribution associated with a particular shift and the associated loss value and the shift in question

    def get_interpretable_params(self, shifts=None, *args, **kwargs):
        if shifts is None:
            shifts = self.matched_shift # if no shift specified we take the best shift
        # init from super
        d = super().get_interpretable_params()
        # select correct parameters in case of separate parameters
        if hasattr(self, "separate_parameters") and self.separate_parameters: # That is, if we select a shift, then retrieve the associated parameters (specified as an argument).
            path_log_params = self.path_log_params[np.arange(self.n_data), :, shifts]
            log_interval_step_weights = self.log_interval_step_weights[np.arange(self.n_data), :, shifts]
        else: # else usual
            log_interval_step_weights = self.log_interval_step_weights
            path_log_params = self.path_log_params
        # compute and add normalised weights
        weight_sum = log_interval_step_weights.exp().sum(dim=1).data.numpy()
        weights = log_interval_step_weights.exp().data.numpy() / weight_sum[:, None] # We normalise the parameter values after delogarithming them, which gives us a new vector of positive integer weights.
        d = dict(**d, weights=weights)
        # format path parameters
        #Here, the learned log parameters are transformed into real parameters, according to the chosen law.
        if self.path_dist == Poisson:
            d = dict(**d, rate=path_log_params.exp().data.numpy().copy())
        elif self.path_dist == Geometric:
            d = dict(**d, probs=path_log_params.sigmoid().data.numpy().copy())
        elif self.path_dist == Gamma:
            d = dict(**d, concentration=path_log_params[:, 0].exp().data.numpy().copy(),
                     rate=path_log_params[:, 1].exp().data.numpy().copy())
        elif self.path_dist in [Binomial, NegativeBinomial]:
            d = dict(**d,
                     total_count=path_log_params[:, 0].exp().data.numpy().copy(),
                     probs=path_log_params[:, 1].sigmoid().data.numpy().copy())
        else:
            raise RuntimeWarning("Failed Case")
        return d  # dictionaries of weights and number of paths calculated

    def perform_diffusion(self):
        # float offset to n (hack e.g. for Gamma, which is not defined for n=0)
        if self.path_dist == Gamma:
            n_offset = 1e-50
        else:
            n_offset = 0
        # uniform offset of probability distribution to ensure finite KL divergence if model produces strict zero
        # probabilities for some pitch classes otherwise (e.g. for Binomial)
        if self.path_dist == Binomial:
            epsilon = 1e-50
        else:
            epsilon = 0
        # path length distribution
        if self.path_dist == Poisson:
            # for new models
            # path_length_dist = Poisson(rate=self.path_log_params.exp()[:, 0])
            # for old models
            path_length_dist = Poisson(rate=self.path_log_params.exp())
        elif self.path_dist == Geometric:
            path_length_dist = Geometric(probs=self.path_log_params.sigmoid()[:, 0])
        elif self.path_dist == Gamma:
            path_length_dist = Gamma(concentration=self.path_log_params[:, 0].exp(),
                                     rate=self.path_log_params[:, 1].exp())
        elif self.path_dist == NegativeBinomial:
            total_count = self.path_log_params[:, 0].exp()
            probs = self.path_log_params[:, 1].sigmoid()
            path_length_dist = NegativeBinomial(total_count=total_count, probs=probs)

        elif self.path_dist == Binomial:
            # Extract log parameters: (n_data, 2)
            total_count = self.path_log_params[:, 0].exp()  #  number of tests (actual, positive)
            total_count_floor = total_count.floor()
            total_count_ceil = total_count.ceil()
            alpha = total_count - total_count_floor         # continuous interpolation
            probs = self.path_log_params[:, 1].sigmoid()    # probability of success

            #floor_bin = Binomial(total_count=total_count_floor, probs=probs)
            #ceil_bin = Binomial(total_count=total_count_ceil, probs=probs)

            #max_n = int(total_count_ceil.max().item())
        else:
            raise RuntimeWarning("Failed Case")
        # callable
        if self.path_dist == Binomial:
            def path_length_dist_func(n):
                # always return a vector of shape (n_data,)
                # build n_tensor on correct device/dtype
                device = total_count_ceil.device
                dtype = total_count_ceil.dtype
                n_val = float(n) if not isinstance(n, torch.Tensor) else float(n.item()) if n.numel() == 1 else None
                if n_val is None:
                    n_tensor = n.to(dtype=dtype, device=device)
                else:
                    n_tensor = torch.full((self.n_data,), n_val, dtype=dtype, device=device)

                # elementwise validity mask: valid where n <= ceil(N)
                valid = n_tensor <= total_count_ceil

                # prepare output vector (zeros by default)
                out = torch.zeros_like(n_tensor, dtype=dtype, device=device)

                if valid.any():
                    # compute k_floor and k_ceil only for valid positions
                    k_floor = torch.minimum(n_tensor, total_count_floor)   # k relative to floor(N)
                    k_ceil  = torch.minimum(n_tensor, total_count_ceil)    # k relative to ceil(N)

                    # but we must compute pmf only where valid; build masked tensors
                    # create masked versions where invalid positions won't be used
                    k_floor_masked = torch.where(valid, k_floor, torch.zeros_like(k_floor))
                    k_ceil_masked  = torch.where(valid, k_ceil, torch.zeros_like(k_ceil))
                    N_floor_masked = torch.where(valid, total_count_floor, torch.ones_like(total_count_floor))  # avoid lgamma(0) oddities
                    N_ceil_masked  = torch.where(valid, total_count_ceil, torch.ones_like(total_count_ceil))

                    # compute logpmf manually (values for invalid positions are arbitrary but will be masked)
                    log_p_floor = binom_logprob_manual_kNk(k_floor_masked, N_floor_masked, probs)
                    log_p_ceil  = binom_logprob_manual_kNk(k_ceil_masked, N_ceil_masked, probs)

                    p_floor = torch.exp(log_p_floor)
                    p_ceil  = torch.exp(log_p_ceil)

                    # interpolation alpha per-piece
                    out = alpha * p_ceil + (1.0 - alpha) * p_floor

                    # ensure positions where valid==False get 0
                    out = torch.where(valid, out, torch.zeros_like(out))

                return out  # shape (n_data,)
            
            #max_n = int(total_count_ceil.max().item())
            cum_path_length_prob = torch.zeros(self.n_data, dtype=torch.float64)
        else:
            def path_length_dist_func(n):
                return path_length_dist.log_prob(n).exp()
        # normalise path length distribution
        if self.path_dist == Gamma: # discresation and normalisation for gamma dist
            l = []
            for n in range(self.max_iterations):
                n = torch.tensor([n + n_offset], dtype=torch.float64)
                l.append(path_length_dist_func(n))
            path_length_dist_arr = torch.stack(l)
            normalisation = path_length_dist_arr.sum(dim=0, keepdim=True)
            assert not np.any(np.isclose(normalisation.data.numpy(), 0)), normalisation.data.numpy().tolist() # so that gamma is non zero
            path_length_dist_arr = path_length_dist_arr / normalisation
        # cumulative sum to track convergence
        cum_path_length_prob = None
        # get interval step probabilities
        interval_step_log_probs = self.log_interval_step_weights - \
                                  self.log_interval_step_weights.logsumexp(dim=1, keepdim=True) # softmax for interval steps
        # initialise running and output distributions
        running_interval_class_distribution = self.init_interval_class_distribution
        self.interval_class_distribution = torch.zeros_like(self.init_interval_class_distribution) # accumulator that will contain the sum, initialised here to zero
        # marginalise latent variable
        #print("separate_parameters:", self.separate_parameters)

        for n in range(self.max_iterations): # avant range(self.max_iterations)
            # probability to reach this step
            # step prob on calcule p(n) sachnat la distribution
            if self.path_dist == Gamma:
                step_prob = path_length_dist_arr[n]  # afore-calcultation pour gamma
            else:
                step_prob = path_length_dist_func(torch.tensor(n, dtype=torch.float64)) # (n_data,) ou (n_data, nshifts)
            if hasattr(self, "separate_parameters") and self.separate_parameters: # if shifts
                # update output distributions (marginalise path length)
                self.interval_class_distribution = self.interval_class_distribution + \
                                                   step_prob[:, None, :] * running_interval_class_distribution # we add to the sum the probability p(n) * the matrix M(n) with n being the current state of the path 
                # perform transition (marginalise interval classes)
                # intermediate tensor has dimensions:
                # (n_data, n_dist_support, n_dist_support, n_interval_steps, n_shifts) = (data, from, to, interval, shift)
                running_interval_class_distribution = torch.einsum("fti,dis,dfs->dts",
                                                                   self.transition_matrix,
                                                                   interval_step_log_probs.exp(),
                                                                   running_interval_class_distribution) # We update the Markov matrix by multiplying the transition matrix * the weights in delog * the previous state. Here we add the shift element.
            else:
                # update output distributions (marginalise path length)
                self.interval_class_distribution = self.interval_class_distribution + \
                                                   step_prob[:, None] * running_interval_class_distribution
                # perform transition (marginalise interval classes)
                # intermediate tensor has dimensions:
                # (n_data, n_dist_support, n_dist_support, n_interval_steps) = (data, from, to, interval)
                running_interval_class_distribution = torch.einsum("fti,di,df->dt",
                                                                   self.transition_matrix,
                                                                   interval_step_log_probs.exp(),
                                                                   running_interval_class_distribution)
            # update cumulative
            if cum_path_length_prob is None:
                cum_path_length_prob = torch.zeros_like(step_prob) # We update the Markov matrix by multiplying the cumulative sum matrix if the probability of p(n) becomes close to 1 and we are after the minimum number of iterations transition * the weights in delog * the previous state here we add the shift element
            # --- break conditions ---
            """if self.path_dist == Binomial:
                max_n = int(total_count_ceil.max().item())
                if n >= max_n:
                    print(f"break after {n+1} iterations (Binomial support exhausted, cum_prob={cum_path_length_prob.mean().item():.3f})")
                    break"""
            if n >= self.effective_min_iterations and np.allclose(cum_path_length_prob.data.numpy(), 1):
                #print(f"break after {n+1} iterations")
                break
        """else:
            with np.printoptions(threshold=np.inf):
                print(f"cum_path_length_prob: {cum_path_length_prob.data.numpy()}")
                print(f"params: {self.get_params()}") 
            raise RuntimeWarning(f"maximum number of iterations ({self.max_iterations}) reached")"""
        # add epsilon
        self.interval_class_distribution = self.interval_class_distribution + epsilon #adds a small epsilon to avoid distributions with values too close to 0 the hack for binomial gammas, etc.
        #print("Before normalisation:", self.interval_class_distribution.sum(dim=1))
        self.interval_class_distribution = self.interval_class_distribution / \
                                           self.interval_class_distribution.sum(dim=1, keepdim=True)  # Since we have truncated the sum to a maximum number of iterations, we renormalise to obtain probabilities summing to 1.
        if np.any(np.isnan(self.interval_class_distribution.data.numpy())):  # nan error
            print(self.interval_class_distribution)
            raise RuntimeWarning("got nan")
        # in addition
        if self.interval_class_distribution.dim() == 3 and self.interval_class_distribution.shape[1] == 1:
            self.interval_class_distribution = self.interval_class_distribution.squeeze(1)

class FactorModel(DiffusionModel):

    def __init__(self,
                 max_iterations=100,
                 path_dist=Poisson,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.path_dist = path_dist
        self.dist_log_params = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise parameters
        if self.path_dist == Poisson:
            self.dist_log_params.data = torch.zeros((self.n_data, self.n_interval_steps), dtype=torch.float64)
        else:
            self.dist_log_params.data = torch.ones((self.n_data,
                                                    self.n_interval_steps,
                                                    2), dtype=torch.float64)

    def get_interpretable_params(self, *args, **kwargs):
        d = super().get_interpretable_params()
        return dict(**d,
                    dist_params=self.dist_log_params.data.numpy().reshape(self.dist_log_params.shape[0], -1).copy())

    def perform_diffusion(self):
        # initialise distribution
        new_interval_class_distribution = self.init_interval_class_distribution
        # ensure finite DKL
        epsilon = 1e-50
        # apply different interval steps successively
        for interval_idx, step_length in enumerate(self.interval_steps):
            # path length distribution
            if self.path_dist == Poisson:
                path_length_dist = Poisson(rate=self.dist_log_params[:, interval_idx].exp())
                def path_length_dist_func(n):
                    return path_length_dist.log_prob(n).exp()
            elif self.path_dist == Binomial:
                total_count = self.dist_log_params[:, interval_idx, 0].exp()
                total_count_floor = total_count.floor()
                total_count_ceil = total_count.ceil()
                alpha = total_count - total_count_floor
                probs = self.dist_log_params[:, interval_idx, 1].sigmoid()
                floor_bin = Binomial(total_count=total_count_floor, probs=probs)
                ceil_bin = Binomial(total_count=total_count_ceil, probs=probs)
                def path_length_dist_func(n):
                    return alpha * ceil_bin.log_prob(n).exp() + (1 - alpha) * floor_bin.log_prob(n).exp()
            else:
                raise RuntimeWarning("Failed Case")
            # cumulative probability weight for termination condition
            cum_path_length_prob = torch.zeros(self.n_data, dtype=torch.float64)
            # marginalise latent variable
            for n in range(self.max_iterations): # self.max_iterations
                # probability to reach this step
                step_prob = path_length_dist_func(torch.tensor(n, dtype=torch.float64))
                # which values are within bounds for original and shifted distribution
                if n == 0:
                    # new dist for accumulation
                    old_interval_class_distribution = new_interval_class_distribution
                    new_interval_class_distribution = step_prob[:, None] * old_interval_class_distribution
                else:
                    if step_length > 0:
                        orig_slice = slice(n * step_length, None)
                        shifted_slice = slice(None, -n * step_length)
                    else:
                        orig_slice = slice(None, n * step_length)
                        shifted_slice = slice(-n * step_length, None)
                    # update output distributions (marginalise path length)
                    new_interval_class_distribution[:, orig_slice] = \
                        new_interval_class_distribution[:, orig_slice] + \
                        step_prob[:, None] * old_interval_class_distribution[:, shifted_slice]
                # update cumulative
                cum_path_length_prob = cum_path_length_prob + step_prob
                if np.allclose(cum_path_length_prob.data.numpy(), 1):
                    print(f"break after {n+1} iterations")
                    break
            else:
                with np.printoptions(threshold=np.inf):
                    print(f"cum_path_length_prob: {cum_path_length_prob.data.numpy()}")
                    print(f"params: {self.get_params()}")
                raise RuntimeWarning(f"maximum number of iterations ({self.max_iterations}) reached")
        self.interval_class_distribution = new_interval_class_distribution + epsilon
        # normalise to account for border effects and path length cut-off
        self.interval_class_distribution = self.interval_class_distribution / \
                                           self.interval_class_distribution.sum(dim=1, keepdim=True)
        if np.any(np.isnan(self.interval_class_distribution.data.numpy())):
            print(self.interval_class_distribution)
            raise RuntimeWarning("got nan")


class SimpleStaticDistributionModel(TonnetzModel):

    def __init__(self,
                 max_iterations=1000):
        super().__init__()
        self.max_iterations = max_iterations
        self.interval_class_log_distribution = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise distribution with mode at reference center
        self.interval_class_log_distribution.data = torch.from_numpy(
            -((np.arange(self.n_dist_support) - self.reference_center) / self.n_dist_support * 10) ** 2
        )[None, :]

    def get_interpretable_params(self, *args, **kwargs):
        return dict()

    def match_distributions(self):
        self.interval_class_distribution = self.interval_class_log_distribution.exp()
        super().match_distributions()


class StaticDistributionModel(TonnetzModel):

    def __init__(self,
                 n_profiles=1,
                 max_iterations=1000,
                 *args, **kwargs):
        super().__init__(latent_shape=(n_profiles,), *args, **kwargs)
        self.max_iterations = max_iterations
        self.interval_class_log_distribution = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise distribution with mode at reference center
        log_dist = -((np.arange(self.n_dist_support) - self.reference_center) / self.n_dist_support * 10) ** 2
        # add some noise for the different profiles
        all_log_dists = np.random.uniform(-1e-3, 1e-3, (self.n_dist_support, self.latent_shape[0]))
        all_log_dists += log_dist[:, None]
        #self.interval_class_log_distribution.data = torch.from_numpy(all_log_dists[None, :])
        # --- FIX: tile across data dimension (n_data) ---
        tiled = np.tile(all_log_dists[None, :, :], (self.n_data, 1, 1))  # shape: (n_data, n_dist_support, n_profiles)
        # convert to torch and assign
        self.interval_class_log_distribution.data = torch.from_numpy(tiled).to(dtype=torch.float64)
        print("n_data:", self.n_data)
        print("interval_class_log_distribution.shape:", self.interval_class_log_distribution.shape)
        # attendu : (n_data, n_dist_support, n_profiles)
        
    def match_distributions(self):
        self.interval_class_distribution = self.interval_class_log_distribution.exp()
        super().match_distributions()


class GaussianModel:

    def __init__(self):
        self.data = None
        self.mean = None
        self.var = None
        self.distributions = None
        self.loss = None

    def set_data(self, data, weights=None):
        self.data = data
        pos = np.arange(data.shape[1])
        self.mean = (data * pos).sum(axis=1) # mean
        self.var = (data * (pos[None, :] - self.mean[:, None]) ** 2).sum(axis=1)
        self.distributions = np.exp(-(pos[None, :] - self.mean[:, None]) ** 2 / self.var[:, None] / 2) # variance
        self.distributions /= self.distributions.sum(axis=1, keepdims=True)
        self.loss = np.where(self.data == 0,
                             np.zeros_like(self.data),
                             self.data * (np.log(self.data) - np.log(self.distributions))).sum(axis=1)

    def get_results(self):
        """
        Compute model loss and return internal results
        :return: tuple of arrays (matched distribution, loss, center) containing results per data point
        """
        # return internal results
        return (self.distributions.copy(),
                self.loss.copy(),
                self.mean.copy())

    def get_interpretable_params(self):
        return dict(mean=self.mean.copy(), std=np.sqrt(self.var))
class StaticDistributionModel2(TonnetzModel):

    def __init__(self,
                 n_profiles=1,
                 max_iterations=1000,
                 *args, **kwargs):
        super().__init__(latent_shape=(n_profiles,), *args, **kwargs)
        self.max_iterations = max_iterations
        self.interval_class_log_distribution = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        log_dist = -((np.arange(self.n_dist_support) - self.reference_center) / self.n_dist_support * 10) ** 2
        all_log_dists = np.random.uniform(-1e-3, 1e-3, (self.n_dist_support, self.latent_shape[0]))
        all_log_dists += log_dist[:, None]

        # Extend to all pieces (n_data)
        n_data = self.n_data  # defined by TonnetzModel.set_data()
        full_log_dists = np.tile(all_log_dists[None, :, :], (n_data, 1, 1))

        self.interval_class_log_distribution.data = torch.from_numpy(full_log_dists)


    def match_distributions(self):
        # reconstruire la distribution avant appel parent
        self.interval_class_distribution = self.interval_class_log_distribution.exp()
        super().match_distributions()

    # trying to fix errors by redifining the dkl (useless)
    @staticmethod
    def dkl(p, q, dim=None):
        p_exp, q_exp = torch.broadcast_tensors(p, q)
        zeros = torch.zeros_like(p_exp)
        term = torch.where(torch.isclose(p_exp, zeros),
                           zeros,
                           p_exp * (p_exp.log() - q_exp.log()))
        dkl = term.sum(dim=dim)
        if torch.isnan(dkl).any():
            print("NaN in dkl:", p_exp.shape, q_exp.shape)
        return dkl

    @staticmethod
    def dkl_log(log_p, log_q, dim=None):
        log_p_exp, log_q_exp = torch.broadcast_tensors(log_p, log_q)
        zeros = torch.zeros_like(log_p_exp)
        mask = torch.isfinite(log_p_exp)
        term = torch.where(mask,
                           log_p_exp.exp() * (log_p_exp - log_q_exp),
                           zeros)
        dkl = term.sum(dim=dim)
        if torch.isnan(dkl).any():
            print("NaN in dkl_log:", log_p_exp.shape, log_q_exp.shape)
        return dkl
