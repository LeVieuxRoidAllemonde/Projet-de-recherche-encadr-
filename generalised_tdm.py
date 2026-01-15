import numpy as np
import torch.nn as nn
from scipy.fftpack import shift
import torch
from copy import deepcopy
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
        # the tdm is not a true predictive model
        raise RuntimeWarning("This is not a normal Module") 

    def set_data(self, data, weights=None):
        self.iteration = 0
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            self.data = data    
        self.n_data, self.n_interval_classes = data.shape # dat must be reduced to picth classes columns
        # set data weights if specified (to weigth composers equally when calculating the mean loss on a whole corpus)
        if weights is not None:
            self.data_weights = torch.from_numpy(weights)
            self.data_weights = self.data_weights / self.data_weights.sum() # weights each piece equally
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
                     -4,  # major third down
                     6,   # tritone up
                     -6   #tritone down
                 ),
                 tonal_space=tuple(range(-17, 18)),
                 soft_max_posterior=False,
                 separate_parameters=False,
                 *args,
                 **kwargs):

        super().__init__()

        # Interval space I 
        self.interval_steps = np.array(interval_steps)
        self.n_interval_steps = len(interval_steps)

        # Tonal space T 
        self.tonal_space = np.array(tonal_space)
        self.n_tonal_space = len(self.tonal_space)

        # data
        self.n_data = None
        self.n_interval_classes = None

        # Model options
        self.soft_max_posterior = soft_max_posterior
        self.separate_parameters = separate_parameters

        # latent distributions / results
        self.interval_class_distribution = None
        self.log_tonal_weights = torch.nn.Parameter(torch.zeros(self.n_tonal_space, dtype=torch.float64))

        self.data = None
        self.matched_dist = None
        self.matched_loss = None
        self.data_weights = None
        self.neg_log_likes = None

        # bayesian case ?
        if self.soft_max_posterior:
            self.beta = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)

    def _loss(self):
        """Negative log-likelihood marginalised over tonal origins."""
        # computing q_c(t) distributions
        #self.perform_diffusion()

        # comparison with real distributions (of the data) and marginalisation over c
        self.match_distributions2()

        neg_log_like = self.neg_log_likes

        if self.data_weights is None:
            return neg_log_like.mean()
        else:
            return (neg_log_like * self.data_weights).sum() # weighted sum for the global loss

    def match_distributions1(self):
        """Computes the marginal negative log-likelihood per piece by summing over all tonal origins c ∈ T."""

        B, C = self.data.shape 
        T = self.n_tonal_space 

        log_p_c = -np.log(T) # log p(c) = constant (uniform prior)  =  the negative log Hc distribution : uniform over T

        log_terms = torch.empty((B, T), dtype=torch.float64) # initalise the logs for each center and piece

        for idx, c in enumerate(self.tonal_space): # tonal center idx and c line in the matrix
            shift = idx  # explicitation indentation

            q_c = self.interval_class_distribution[:, shift:shift + C]
            q_c = q_c / q_c.sum(dim=1, keepdim=True)
            kl = self.dkl(self.data, q_c, dim=1)

            log_terms[:, idx] = log_p_c - kl

        log_likelihood = torch.logsumexp(log_terms, dim=1)
        self.neg_log_likes = -log_likelihood
    def match_distributions2(self):
        B, C = self.data.shape # number of pieces, number of of tpcs
        T = self.n_tonal_space 
        
        log_p_c = self.log_tonal_weights - self.log_tonal_weights.logsumexp(dim=0) # a softmax

        log_terms = torch.empty((B, T), dtype=torch.float64)

        for idx in range(T):
            #q_c = self.interval_class_distribution[:, idx:idx + C]
            q_c = self.interval_class_distribution[idx][None, :].repeat(B, 1)
            q_c = q_c / q_c.sum(dim=1, keepdim=True)
            #print("dim de q : ", q_c.shape)
            kl = self.dkl(self.data, q_c, dim=1) # loss
            log_terms[:, idx] = log_p_c[idx] - kl # adjusting the weights

        log_likelihood = torch.logsumexp(log_terms, dim=1)
        self.neg_log_likes = -log_likelihood
    def get_interpretable_params(self, *args, **kwargs):
        """
        Returns interpretable parameters of the model: global learned distribution over tonal origins, posterior distribution over tonal origins per piece, negative log-likelihood per piece"""

        # first the distribution of tonal centers
        log_p_c = self.log_tonal_weights
        p_c = (log_p_c - log_p_c.logsumexp(dim=0)).exp() #softmax normalise

        tonal_origin_distribution = {int(t): p_c[i].item() for i, t in enumerate(self.tonal_space)}

        # Posterior distribution by piece p(c | t) fro the neg neg_log_likes

        B = self.n_data
        T = self.n_tonal_space
        C = self.n_interval_classes

        posterior_per_piece = []

        for b in range(B):
            log_terms = torch.empty(T, dtype=torch.float64)

            for i, c in enumerate(self.tonal_space):
                shift = int(c - self.tonal_space[0])

                q_c = self.interval_class_distribution[b, shift:shift + C]
                q_c = q_c / q_c.sum()

                kl = self.dkl(self.data[b:b+1], q_c[None, :], dim=1)[0]

                log_terms[i] = log_p_c[i] - kl

            log_post = log_terms - log_terms.logsumexp(dim=0)
            post = log_post.exp()

            posterior_per_piece.append({int(self.tonal_space[i]): post[i].item()for i in range(T)})

        # putting everything together
        results = dict(tonal_origin_distribution=tonal_origin_distribution, tonal_origin_posterior=posterior_per_piece,neg_log_likelihood=self.neg_log_likes.data.numpy().copy())

        if self.soft_max_posterior:
            results["beta"] = self.beta.exp().item()

        return results
    
class DiffusionModel(TonnetzModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_state = None            # p(τ0 | c) for all tonal center c
        self.transition_matrix = None     # Transitions on the Tonnetz

    def set_data(self, *args, **kwargs):
        """Preparing diffusion-specific structures: initial states p(τ0 | c) for all tonal origins, Tonnetz transition matrix (independent of c)"""
        # load data, set n_data, n_interval_classes, etc.
        super().set_data(*args, **kwargs)
        T = self.n_tonal_space

        init_state = np.zeros((T, T), dtype=np.float64) # Identity matrix
        for idx in range(T):
            init_state[idx, idx] = 1.0

        self.init_state = torch.from_numpy(init_state)
        # the  Tonnetz transition is matrix (T, T, n_interval_steps) ; then transition_matrix[i, j, k] = 1  meaning applying interval_steps[k] one moves from i to j
        transition_matrix = np.zeros((T, T, self.n_interval_steps),dtype=np.float64)

        for k, step in enumerate(self.interval_steps): # initializing the transition matrix with ones
            for i in range(T):
                j = i + step
                if 0 <= j < T:
                    transition_matrix[i, j, k] = 1.0 

        self.transition_matrix = torch.from_numpy(transition_matrix)
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

        # Computing the necessary number of iteratiosn

        if self.min_iterations is None:
            largest_step_down = -min(np.min(self.interval_steps), 0)
            largest_step_up = max(np.max(self.interval_steps), 0)

            # maximal distance within tonal space (everything should be within the bounds of tonal space : we "cut" the edges of the support)
            tonal_min = np.min(self.tonal_space)
            tonal_max = np.max(self.tonal_space)
            max_distance = tonal_max - tonal_min

            # minimal number of iterations to run this distance
            self.effective_min_iterations = int(np.ceil(max(max_distance / largest_step_up if largest_step_up > 0 else 0, max_distance / largest_step_down if largest_step_down > 0 else 0))) + 1
        else:
            self.effective_min_iterations = self.min_iterations

        # interval weights initialisatoo (interval_steps)

        if self.n_interval_steps == 6:
            # advantage given to fifths
            weights = np.array([3, 3, 0, 0, 0, 0], dtype=np.float64)
        else:
            weights = np.zeros(self.n_interval_steps, dtype=np.float64)

        # 
        """full_weights = np.zeros((self.n_data, self.n_interval_steps), dtype=np.float64)
        full_weights[:, :] = weights[None, :]

        self.log_interval_step_weights.data = torch.from_numpy(full_weights)"""
        self.log_interval_step_weights.data = torch.from_numpy(weights)
        # path length initialisatiojn
        if self.path_dist in [Poisson, Geometric]:
            default_params = [1]

        elif self.path_dist in [Gamma, Binomial, NegativeBinomial]:
            if self.path_dist == Gamma:
                default_params = [2, -2]
            elif self.path_dist == Binomial:
                default_params = [2, 0]
            else:
                default_params = [0, 0]
        else:
            raise RuntimeError("Unsupported path distribution")

        default_params = np.array(default_params, dtype=np.float64)

        full_params = np.zeros((self.n_data, len(default_params)), dtype=np.float64)
        full_params[:, :] = default_params[None, :]

        self.path_log_params.data = torch.from_numpy(full_params)
    def _loss(self):
        self.perform_diffusion() # computes p(t | c, w, λ)
        return super()._loss() # comparison to data

    def perform_diffusion(self):
        """Computes p(t | c, w, λ) for all tonal centres c by marginalising over path length and latent Tonnetz trajectories (Algorithm 1)."""

        if self.path_dist == Gamma: # a few hacks
            n_offset = 1e-50
        else:
            n_offset = 0.0

        epsilon = 1e-50 if self.path_dist == Binomial else 0.0
        
        #Path length distribution p(n | λ)
        if self.path_dist == Poisson:
            path_length_dist = Poisson(rate=self.path_log_params.exp())

            def path_length_prob(n):
                return path_length_dist.log_prob(torch.tensor(n)).exp()

        elif self.path_dist == Geometric:
            path_length_dist = Geometric(probs=self.path_log_params.sigmoid())

            def path_length_prob(n):
                return path_length_dist.log_prob(torch.tensor(n)).exp()

        elif self.path_dist == Gamma:
            path_length_dist = Gamma(concentration=self.path_log_params[:, 0].exp(),rate=self.path_log_params[:, 1].exp())

            probs = []
            for n in range(self.max_iterations):
                probs.append(path_length_dist.log_prob(torch.tensor(n + n_offset)).exp())
            path_length_probs = torch.stack(probs, dim=0)
            path_length_probs /= path_length_probs.sum(dim=0, keepdim=True)

            def path_length_prob(n):
                return path_length_probs[n]

        elif self.path_dist == Binomial:
            #total_count = self.path_log_params[:, 0].exp()
            total_count = torch.clamp(self.path_log_params[:, 0].exp(),max=40)

            total_count_floor = total_count.floor()
            total_count_ceil = total_count.ceil()
            alpha = total_count - total_count_floor
            probs = self.path_log_params[:, 1].sigmoid()

            def path_length_prob(n):
                n = torch.tensor(float(n))
                valid = n <= total_count_ceil
                out = torch.zeros_like(total_count)

                if valid.any():
                    kf = torch.minimum(n, total_count_floor)
                    kc = torch.minimum(n, total_count_ceil)

                    log_pf = binom_logprob_manual_kNk(kf, total_count_floor, probs)
                    log_pc = binom_logprob_manual_kNk(kc, total_count_ceil, probs)

                    pf = log_pf.exp()
                    pc = log_pc.exp()

                    out = alpha * pc + (1 - alpha) * pf
                    out = torch.where(valid, out, torch.zeros_like(out))

                return out

        else:
            raise RuntimeError("Unsupported path distribution")

        # Interval transition probabilities p(τ' | τ, w)
        #interval_step_log_probs = (self.log_interval_step_weights - self.log_interval_step_weights.logsumexp(dim=0, keepdim=True))

        interval_step_probs = interval_step_probs.mean(dim=1)
        #print(interval_step_probs.shape)

        # Initialisation (Algorithm 1, lines 1–3)
        v = self.init_state.clone()  # p(τ0 | c)
        u = torch.zeros_like(v)   # accumulator

        cum_prob = 0.0
        
        # 4. Main diffusion loop (Algorithm 1, lines 4–7)
        for n in range(self.max_iterations):
            pn = path_length_prob(n)

            # (5) u ← u + p(n|λ) · v
            u += pn[:, None] * v if pn.ndim == 1 else pn * v

            cum_prob += pn

            # (6) v ← M v
            """v = torch.einsum(
                "tfi,i,ct->cf",
                self.transition_matrix,
                interval_step_probs,
                v
            )"""
            M = torch.einsum("tfi,i->tf",self.transition_matrix, interval_step_probs)
            v = v @ M
            # (7) adaptive stopping
            if n >= self.effective_min_iterations:
                if torch.allclose(cum_prob, torch.ones_like(cum_prob), atol=1e-5):
                    break

        u = u + epsilon
        u = u / u.sum(dim=1, keepdim=True)

        self.interval_class_distribution = u
    def get_results(self):
        """ Returns interpretable results of the TonnetzModel"""
        self._loss()

        B, C = self.data.shape
        T = self.n_tonal_space

        #  Prior over tonal origins p(c) 
        log_p_c = self.log_tonal_weights - self.log_tonal_weights.logsumexp(dim=0)
        p_c = log_p_c.exp()

        tonal_origin_prior = {int(self.tonal_space[i]): p_c[i].item()for i in range(T)}

        # Posterior per piece p(c | t_b)
        tonal_origin_posterior = []
        tonal_origin_map = []

        for b in range(B):
            log_terms = torch.empty(T, dtype=torch.float64)

            for idx in range(T):
                # T == C case : direct indexing
                q_c = self.interval_class_distribution[idx][None, :]  # (1, C)
                kl = self.dkl(self.data[b:b+1],q_c,dim=1)[0]
                log_terms[idx] = log_p_c[idx] - kl

            log_post = log_terms - log_terms.logsumexp(dim=0)
            post = log_post.exp()

            posterior_dict = {int(self.tonal_space[i]): post[i].item()for i in range(T)}

            tonal_origin_posterior.append(posterior_dict)
            tonal_origin_map.append(int(self.tonal_space[post.argmax().item()]))

        results = dict(
            loss=self.neg_log_likes.data.numpy().copy(),
            predicted_distribution = self.interval_class_distribution.detach().cpu().numpy(),
            tonal_origin_prior=tonal_origin_prior,
            tonal_origin_posterior=tonal_origin_posterior,
            tonal_origin_map=tonal_origin_map
        )

        if self.soft_max_posterior:
            results["beta"] = self.beta.exp().item()

        return results
    def get_interpretable_params(self):
        """Return interpretable, piece-level parameters for the multi-centre TDM."""
        self._loss()

        B = self.n_data
        T = self.n_tonal_space

        results = []

        # prior distribution over tonal centers
        log_p_c = self.log_tonal_weights - self.log_tonal_weights.logsumexp(dim=0)

        for b in range(B):
            # normalized interval step weights per piece
            piece_dict = {}
            w = self.log_interval_step_weights[b].exp()
            w_norm = (w / w.sum()).detach().cpu().numpy()

            piece_dict["interval_step_weights"] = w_norm
            #Path length distribution parameters (per piece)
            if self.path_dist == Poisson:piece_dict["path_length_distribution"] = {"type": "Poisson","rate": self.path_log_params[b].exp().item()}

            elif self.path_dist == Binomial:
                piece_dict["path_length_distribution"] = {"type": "Binomial","total_count": self.path_log_params[b, 0].exp().item(),"probs": self.path_log_params[b, 1].sigmoid().item()}

            elif self.path_dist == Gamma:
                piece_dict["path_length_distribution"] = {"type": "Gamma","concentration": self.path_log_params[b, 0].exp().item(),"rate": self.path_log_params[b, 1].exp().item()}

            else:
                raise RuntimeError("Unsupported path_dist in get_interpretable_params")

            # Tonal origin posterior p(c | t_b)
            log_terms = torch.empty(T, dtype=torch.float64)

            for idx in range(T):
                q_c = self.interval_class_distribution[b, idx]
                kl = self.dkl(self.data[b:b+1],q_c[None, :],dim=1)[0]
                log_terms[idx] = log_p_c[idx] - kl

            log_post = log_terms - log_terms.logsumexp(dim=0)
            post = log_post.exp().detach().cpu().numpy()

            piece_dict["tonal_origin_posterior"] = {int(self.tonal_space[i]): float(post[i])for i in range(T)}
            results.append(piece_dict)

        return {"pieces": results}
