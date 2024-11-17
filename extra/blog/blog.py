# self contained code for blog post
import math
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyscf
import pyscf.cc
from tqdm import tqdm


def local_energy(wavefunction, atoms, charges, pos):
    return kinetic_energy(wavefunction, pos) + potential_energy(atoms, charges, pos)


def kinetic_energy(wavefunction, pos):
    """Kinetic energy term of Hamiltonian"""
    laplacian = jnp.trace(jax.hessian(wavefunction)(pos))
    return -0.5 * laplacian / wavefunction(pos)


def potential_energy(atoms, charges, pos):
    """Potential energy term of Hamiltonian"""
    pos = pos.reshape(-1, 3)

    r_ea = jnp.linalg.norm(pos[:, None, :] - atoms[None, :, :], axis=-1)

    i, j = jnp.triu_indices(pos.shape[0], k=1)
    r_ee = jnp.linalg.norm(pos[i] - pos[j], axis=-1)

    i, j = jnp.triu_indices(atoms.shape[0], k=1)
    r_aa = jnp.linalg.norm(atoms[i] - atoms[j], axis=-1)
    z_aa = charges[i] * charges[j]

    v_ee = jnp.sum(1 / r_ee)
    v_ea = -jnp.sum(charges / r_ea)
    v_aa = jnp.sum(z_aa / r_aa)
    return v_ee + v_ea + v_aa


def wavefunction_h(pos):
    return jnp.exp(-jnp.linalg.norm(pos))


atoms = np.array([[0.0, 0.0, 0.0]])
charges = np.array([1.0])
pos = np.random.randn(3)  # randomly sample electron position
print(local_energy(wavefunction_h, atoms, charges, pos))




@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None, None, 0))
def metropolis(
    wavefunction: Callable,
    pos: jax.Array,
    step_size: float,
    mcmc_steps: int,
    key: jax.Array,
):
    """MCMC step

    Args:
        wavefunction: neural wavefunction
        pos: [3N] current electron positions flattened
        step_size: std of proposal for metropolis sampling
        mcmc_steps: number of steps to perform
        key: random key
    """

    def step(_, carry):
        pos, prob, num_accepts, key = carry
        key, subkey = jax.random.split(key)
        pos_proposal = pos + step_size * jax.random.normal(subkey, shape=pos.shape)
        prob_proposal = wavefunction(pos_proposal) ** 2

        key, subkey = jax.random.split(key)
        accept = jax.random.uniform(subkey) < prob_proposal / prob
        prob = jnp.where(accept, prob_proposal, prob)
        pos = jnp.where(accept, pos_proposal, pos)
        num_accepts = num_accepts + jnp.sum(accept)

        return pos, prob, num_accepts, key

    prob = wavefunction(pos) ** 2
    carry = (pos, prob, 0, key)
    pos, prob, num_accepts, key = jax.lax.fori_loop(0, mcmc_steps, step, carry)
    return pos, num_accepts / mcmc_steps


# plot effect of mcmc on hydrogen wavefunction
# probably not going to be included in blog post
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

pos = 0.4 * jax.random.normal(jax.random.key(0), shape=(4096, 3))
r = np.linspace(0, 5, 50)
fig, ax = plt.subplots(figsize=(5, 4))

def update(frame):
    global pos
    keys = jnp.array(jax.random.split(jax.random.key(frame), 4096))
    ax.clear()
    ax.plot(r, 4 * r**2 * jnp.exp(-2 * r), label=r"$4r^2e^{-2r}$", c="k")
    ax.hist(np.linalg.norm(pos, axis=-1).ravel(), bins=r, histtype="step", density=True)
    ax.legend(loc="upper right")
    ax.set_title(f"Radial Distribution (Step = {frame})")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel("Density")
    pos, _ = metropolis(wavefunction_h, pos, 0.5, 1, keys)
    return []

ani = FuncAnimation(fig, update, frames=np.arange(0, 40, 1), blit=False)
ani.save("mcmc.gif", writer="imagemagick", dpi=200)
plt.close(fig)



def make_loss(atoms, charges):
    # Based on implementation in https://github.com/google-deepmind/ferminet/

    @eqx.filter_custom_jvp
    def total_energy(wavefunction, pos):
        """Define L()"""
        batch_local_energy = jax.vmap(local_energy, (None, None, None, 0))
        e_l = batch_local_energy(wavefunction, atoms, charges, pos)
        loss = jnp.mean(e_l)
        return loss, e_l

    @total_energy.def_jvp
    def total_energy_jvp(primals, tangents):
        """Define gradient of L()"""
        wavefunction, pos = primals
        log_wavefunction = lambda psi, pos: jnp.log(psi(pos))
        batch_wavefunction = jax.vmap(log_wavefunction, (None, 0))
        psi_primal, psi_tangent = eqx.filter_jvp(batch_wavefunction, primals, tangents)
        loss, local_energy = total_energy(wavefunction, pos)
        primals_out = loss, local_energy
        batch_size = jnp.shape(local_energy)[0]
        tangents_out = (jnp.dot(psi_tangent, local_energy - loss) / batch_size, local_energy)
        return primals_out, tangents_out

    return total_energy


class Linear(eqx.Module):
    """Linear layer"""

    weights: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        lim = math.sqrt(1 / (in_size + out_size))
        self.weights = jax.random.uniform(key, (in_size, out_size), minval=-lim, maxval=lim)
        self.bias = jnp.zeros(out_size)

    def __call__(self, x):
        return jnp.dot(x, self.weights) + self.bias


class PsiMLP(eqx.Module):
    """Simple MLP-based model using Slater determinant"""

    spins: tuple[int, int]
    linears: list[Linear]
    orbitals: Linear
    sigma: jax.Array 
    pi: jax.Array

    def __init__(
        self,
        hidden_sizes: list[int],
        spins: tuple[int, int],
        determinants: int,
        key: jax.Array,
    ):
        num_atoms = 1  # assume one atom
        sizes = [5] + hidden_sizes  # 5 input features
        key, *keys = jax.random.split(key, len(sizes))
        self.linears = []
        for i in range(len(sizes) - 1):
            self.linears.append(Linear(sizes[i], sizes[i + 1], keys[i]))
        self.orbitals = Linear(sizes[-1], sum(spins) * determinants, key)
        self.sigma = jnp.ones((num_atoms, sum(spins) * determinants))
        self.pi = jnp.ones((num_atoms, sum(spins) * determinants))
        self.spins = spins

    def __call__(self, pos):
        # atom electron displacement [electron, atom, 3]
        ae = pos.reshape(-1, 1, 3)
        # atom electron distance [electron, atom, 1]
        r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
        # feature for spins; 1 for up, -1 for down [atom, 1]
        spins = jnp.concatenate([jnp.ones(self.spins[0]), jnp.ones(self.spins[1]) * -1])

        # combine into features
        h = jnp.concatenate([ae, r_ae], axis=2)
        h = h.reshape([h.shape[0], -1])
        h = jnp.concatenate([h, spins[:, None]], axis=1)

        # multi-layer perceptron with tanh activations
        for linear in self.linears:
            h = jnp.tanh(linear(h))

        phi = self.orbitals(h) * jnp.sum(self.pi * jnp.exp(-self.sigma * r_ae), axis=1)

        # [electron, electron * determinants] -> [determinants, electron, electron]
        phi = phi.reshape(phi.shape[0], -1, phi.shape[0]).transpose(1, 0, 2)
        det = jnp.linalg.det(phi)
        return jnp.sum(det)



def vmc(
    wavefunction: Callable,
    atoms: jax.Array,
    charges: jax.Array,
    spins: tuple[int, int],
    *,
    batch_size: int = 4096,
    mcmc_steps: int = 50,
    warmup_steps: int = 200,
    init_width: float = 0.4,
    step_size: float = 0.2,
    learning_rate: float = 3e-3,
    iterations: int = 2000,
    key: jax.Array,
):
    """Perform VMC

    Args:
        wavefunction: neural wavefunction
        atoms: [M, 3] atomic positions
        charges: [M] atomic charges
        spins: number spin-up, spin-down electrons
        batch_size: number of electron configurations to sample
        mcmc_steps: number of mcmc steps to perform between neural network
            updates (lessens autocorrelation)
        warmup_steps: number of mcmc steps to perform before starting training
        step_size: std of proposal for metropolis sampling
        learning_rate: learning rate
        iterations: number of neural network updates
        key: random key
    """
    total_energy = make_loss(atoms, charges)

    # initialize electron positions and perform warmup mcmc steps
    key, subkey = jax.random.split(key)
    pos = init_width * jax.random.normal(subkey, shape=(batch_size, sum(spins) * 3))
    key, *subkeys = jax.random.split(key, batch_size + 1)
    pos, _ = metropolis(wavefunction, pos, step_size, warmup_steps, jnp.array(subkeys))

    # Adam optimizer with gradient clipping
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(eqx.filter(wavefunction, eqx.is_array))

    @eqx.filter_jit
    def train_step(wavefunction, pos, key, opt_state):
        key, *subkeys = jax.random.split(key, batch_size + 1)
        pos, accept = metropolis(wavefunction, pos, step_size, mcmc_steps, jnp.array(subkeys))
        (loss, _), grads = eqx.filter_value_and_grad(total_energy, has_aux=True)(wavefunction, pos)
        updates, opt_state = optimizer.update(grads, opt_state, wavefunction)
        wavefunction = eqx.apply_updates(wavefunction, updates)
        return wavefunction, pos, key, opt_state, loss, accept

    losses, pmoves = [], []
    pbar = tqdm(range(iterations))
    for _ in pbar:
        wavefunction, pos, key, opt_state, loss, pmove = train_step(wavefunction, pos, key, opt_state)
        pmove = pmove.mean()
        losses.append(loss)
        pmoves.append(pmove)
        pbar.set_description(f"Energy: {loss:.4f}, P(move): {pmove:.2f}")

    return losses, pmoves



# Lithium at origin
atoms = jnp.zeros((1, 3))
charges = jnp.array([3.0])
spins = (2, 1) # 2 spin-up, 1 spin-down electrons

key = jax.random.key(0)
key, subkey = jax.random.split(key)
model = PsiMLP(hidden_sizes=[64, 64, 64], determinants=4, spins=spins, key=key)
losses, _ = vmc(model, atoms, charges, spins, key=subkey)

def smooth(losses, window_pct=10):
    # smooth with median of last 10% of samples
    window = int(len(losses) * window_pct / 100)
    return [np.median(losses[max(0, i-window):i+1]) for i in range(len(losses))]

# Smoothed loss from model
plt.figure(figsize=(5, 3), dpi=300)
plt.plot(smooth(losses), label="MLP")

# CCSD(T) calculation
basis = "CC-pV5Z"
m = pyscf.gto.mole.M(atom="Li 0 0 0", basis=basis, spin=1)
mf = pyscf.scf.RHF(m).run()
mycc = pyscf.cc.CCSD(mf).run()
et_correction = mycc.ccsd_t()
e_tot = mycc.e_tot + et_correction
plt.axhline(e_tot, c="k", ls="--", label=f"CCSD(T)/{basis}", zorder=3)

# Exact from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.47.3649
plt.axhline(-7.47806032, c="k", label="Exact Lithium energy", zorder=3)
plt.legend()
plt.ylim(-7.5, -7.3)
plt.xlabel("Iteration")
plt.ylabel("Energy (a.u.)")
plt.tight_layout()
plt.savefig("li_psimlp.png")



class PsiMLPJastrow(PsiMLP):

    beta: jax.Array  # parameter for Jastrow

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = jnp.array(1.0)

    def __call__(self, pos):
        det = super().__call__(pos)
        pos = pos.reshape(-1, 3)
        i, j = jnp.triu_indices(pos.shape[0], k=1)
        r_ee = jnp.linalg.norm(pos[i] - pos[j], axis=1)
        alpha = jnp.where((i < self.spins[0]) == (j < self.spins[0]), 0.25, 0.5)
        jastrow = jnp.exp(jnp.sum(alpha * r_ee / (1.0 + self.beta * r_ee)))
        return det * jastrow


model = PsiMLPJastrow(hidden_sizes=[64, 64, 64], determinants=4, spins=spins, key=key)
losses, _ = vmc(model, atoms, charges, spins, key=subkey)
plt.plot(smooth(losses), label="MLP + Jastrow")
plt.legend()
plt.tight_layout()
plt.savefig("li_psimlp_jastrow.png")
