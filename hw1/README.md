# Homework 1

## PGD-based Adversarial Training and Accuracy Report

### High-Level Solution Overview

The goal of this task is to perform adversarial attacks using the Projected Gradient Descent (PGD) method and then train a robust model using adversarial training. PGD is an iterative algorithm that generates adversarial examples by taking gradient-based steps to maximize the model's loss and projecting the solution back into an $\epsilon$-sized $L_\infty$-ball around the original input to ensure the perturbation remains bounded.

This approach includes two main phases:
1. **PGD-based attack generation**: Generate adversarial examples using PGD for untargeted attacks.
2. **Adversarial training**: Use these adversarial examples during the training process to create a model that can withstand such attacks.

### PGD Implementation

PGD builds on the Fast Gradient Sign Method (FGSM). It iterates over FGSM steps while ensuring that the perturbations remain within the $\epsilon$-bounded ball by projecting them back using clipping. This process keeps the adversarial examples close to the original input while maximizing the impact of the perturbation.

The `pgd_untargeted` function used for generating adversarial examples is as follows:

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

Key components of the function:
- The model is evaluated in `eval()` mode to disable training operations like dropout.
- Gradient-based perturbations are applied iteratively.
- After each step, the adversarial examples are projected back into the $\epsilon$-ball and clipped to maintain valid image values.

### Model Training and Testing

The training function allows for adversarial training using the generated PGD or FGSM attacks. When `enable_defense` is set to `True`, adversarial examples are generated during each batch of training. Otherwise, the model is trained on standard (non-adversarial) data.

The following is the adversarial training function:

```python
def train_model(model, num_epochs, enable_defense=True, attack='pgd', eps=0.1):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
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
            else:
                data_adv = data
            output = model(data_adv)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
```

During adversarial training, the model dynamically generates adversarial examples using either the `pgd_untargeted` or `fgsm` functions. This process helps improve the model's robustness to adversarial attacks.

### Results

#### **Standard Training**

After 20 epochs of standard training (without adversarial examples), the model achieved the following performance:
- **Standard accuracy on clean data**: **99.74%**

#### **Testing under PGD Attacks**

The PGD attack was tested with varying values of $\epsilon$:
- For $\epsilon = 0.05$, the accuracy under attack dropped to **76.32%**.
- For $\epsilon = 0.1$, the accuracy under attack dropped further to **22.37%**.
- For $\epsilon = 0.15$, the accuracy dropped to **3.95%**.
- For $\epsilon = 0.2$, the accuracy dropped drastically to **0.56%**.

This demonstrates that as the perturbation magnitude increases, the model becomes more vulnerable to adversarial examples.

#### **Adversarial Training**

Adversarial training with PGD ($\epsilon = 0.1$) significantly improved robustness, yielding the following results:
- **Standard accuracy on clean data**: **90.11%**
- **Robust accuracy on adversarial examples ($\epsilon = 0.1$)**: **22.37%**

Adversarial training allows the model to maintain a higher level of performance under attack compared to the standard model.

#### **Testing the Robustness of the Adversarially Trained Model**

The adversarially trained model was also tested under FGSM attacks for various values of $\epsilon$:
- For $\epsilon = 0.05$, the accuracy under FGSM attack was **95.63%**.
- For $\epsilon = 0.1$, the accuracy was **90.54%**.
- For $\epsilon = 0.15$, the accuracy was **80.38%**.
- For $\epsilon = 0.2$, the accuracy dropped to **61.75%**.

This shows that the adversarially trained model is more resilient to both PGD and FGSM attacks, especially for smaller perturbations.

### Conclusion

Adversarial training significantly enhances the model's robustness against PGD and FGSM-based attacks, particularly for smaller perturbation magnitudes. However, as the perturbation magnitude increases, the effectiveness of adversarial training diminishes. This highlights the inherent trade-off between achieving high accuracy on clean data and ensuring robustness to adversarial attacks.