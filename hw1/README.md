# Homework 1

## PGD-based Adversarial Training and Accuracy Report

### High-Level Solution Overview

The goal of the task is to perform adversarial attacks using the Projected Gradient Descent (PGD) method and subsequently train a robust model using adversarial training. PGD is an iterative algorithm used to generate adversarial examples by taking steps based on the gradients of the model's loss function and projecting the solution back into an ε-sized \(L_\infty\)-ball around the original input to ensure the perturbation remains bounded.

This approach includes two main phases:
1. **PGD-based attack generation**: Create adversarial examples using PGD for untargeted attacks.
2. **Adversarial training**: Incorporate these adversarial examples during the training process to build a model that can withstand such attacks.

### PGD Implementation

PGD is built upon the Fast Gradient Sign Method (FGSM). It iterates over FGSM steps and ensures that the perturbations remain within the ε-bounded ball by projecting them back using clipping. This keeps the adversarial examples close to the original input while maximizing the impact of the perturbation.

PGD function:

```python
def pgd_untargeted(model, x, y, *, k=7, eps=0.1, eps_step=0.025):
    model.eval()
    x_adv = x.clone().detach()
    x_orig = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + eps_step * grad_sign
        eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + eta
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv
```

In the function:
- The model is evaluated in `eval()` mode.
- Gradient-based perturbations are applied iteratively.
- After each step, the adversarial examples are projected back into the ε-ball and clipped to maintain valid image values.

### Model Training and Testing

The training function includes a toggle for adversarial training (defense). When `enable_defense` is set to `True`, adversarial examples are generated using PGD or FGSM and used for training. Otherwise, the model is trained on the standard data.


Train function:
```python
def train_model(model, num_epochs, enable_defense=True, attack='pgd', eps=0.1):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if enable_defense:
                if attack == 'fgsm':
                    data_adv = fgsm(model, data, target, eps=eps)
                elif attack == 'pgd':
                    data_adv = pgd_untargeted(model, data, target, k=7, eps=eps, eps_step=eps/4)
            else:
                data_adv = data
            output = model(data_adv)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
```

The defense mechanism uses either `pgd_untargeted` or `fgsm` to generate adversarial examples dynamically during training, making the model robust to such attacks.

### Results

**Standard Training**
- After 20 epochs, the model achieved an accuracy of **99.74%** on clean data.

**Testing under PGD Attacks**
- For ε = 0.05, the accuracy under attack dropped to **73.20%**.
- For ε = 0.1, the accuracy under attack further dropped to **20.69%**.
- For ε = 0.2, the model was almost entirely compromised, with accuracy dropping to **0.18%**.

**Adversarial Training**
Adversarial training with PGD significantly improved robustness, yielding a final accuracy of **90.28%** on clean data.

This shows that adversarial training strengthens the model against attacks compared to standard training, especially for larger values of epsilon.

### Conclusion

Adversarial training helps mitigate vulnerabilities to PGD-based attacks, but the effectiveness decreases as the perturbation magnitude (ε) increases.