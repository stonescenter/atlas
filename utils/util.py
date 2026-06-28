import torch
import torch.nn.functional as F

def gradient_direction_stats(model, walk_loss, link_loss):
    params = [p for p in model.parameters() if p.requires_grad]

    walk_grads = torch.autograd.grad(
        walk_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    link_grads = torch.autograd.grad(
        link_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )

    walk_parts = []
    link_parts = []

    for walk_grad, link_grad in zip(walk_grads, link_grads):
        if walk_grad is None or link_grad is None:
            continue

        walk_parts.append(walk_grad.detach().reshape(-1))
        link_parts.append(link_grad.detach().reshape(-1))

    if not walk_parts:
        return {
            "cosine": float("nan"),
            "walk_norm": 0.0,
            "link_norm": 0.0,
        }

    walk_vec = torch.cat(walk_parts)
    link_vec = torch.cat(link_parts)

    cosine = F.cosine_similarity(
        walk_vec,
        link_vec,
        dim=0,
        eps=1e-12,
    ).item()

    return {
        "cosine": cosine,
        "walk_norm": walk_vec.norm().item(),
        "link_norm": link_vec.norm().item(),
    }
