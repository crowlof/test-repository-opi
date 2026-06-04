# Решение задания Q1: Градиент MSE для линейной регрессии

## 1. Вывод частной производной $ \frac{\partial Q}{\partial \theta_j} $

Дана функция потерь (эмпирический риск):

$$
Q(\theta, X^\ell) = \frac{1}{\ell} \sum_{i=1}^{\ell} \bigl( \langle \theta, x_i \rangle + \theta_0 - y_i \bigr)^2,
$$

где  
$\theta = (\theta_1,\dots,\theta_d)^\top$ – вектор весов,  
$\theta_0$ – смещение (bias),  
$x_i = (x_{i1},\dots,x_{id})^\top$ – вектор признаков $i$-го объекта,  
$y_i$ – целевое значение.

Возьмём частную производную по $\theta_j$ ($j=1,\dots,d$):

1. **Линейность дифференцирования** (производная суммы, вынос константы $1/\ell$):
   $$
   \frac{\partial Q}{\partial \theta_j} = \frac{1}{\ell} \sum_{i=1}^{\ell} \frac{\partial}{\partial \theta_j} \bigl( \underbrace{\langle \theta, x_i \rangle + \theta_0 - y_i}_{=: \varepsilon_i} \bigr)^2.
   $$

3. **Производная квадрата** (правило цепочки):
   $$
   \frac{\partial}{\partial \theta_j} \varepsilon_i^2 = 2\varepsilon_i \cdot \frac{\partial \varepsilon_i}{\partial \theta_j}.
   $$

4. **Вычисление $\frac{\partial \varepsilon_i}{\partial \theta_j}$**:
   $$
   \varepsilon_i = \sum_{k=1}^{d} \theta_k x_{ik} + \theta_0 - y_i \quad\Rightarrow\quad \frac{\partial \varepsilon_i}{\partial \theta_j} = x_{ij}.
   $$

5. **Подстановка**:
   $$
\frac{\partial Q}{\partial \theta_j} = \frac{2}{\ell} \sum_{i=1}^{\ell} \bigl( \langle \theta, x_i \rangle + \theta_0 - y_i \bigr) x_{ij}
$$
   
---

## 2. Матричная форма градиента по всему вектору $\theta$

Введём матричные обозначения:  
- $X$ – матрица размера $\ell \times d$ ($i$-я строка = $x_i^\top$),  
- $\mathbf{y} = (y_1,\dots,y_\ell)^\top$,  
- $\mathbf{1}$ – вектор-столбец из единиц длины $\ell$.

Тогда вектор ошибок:
$$
\varepsilon = X\theta + \theta_0\mathbf{1} - \mathbf{y}.
$$

Функционал:
$$
Q = \frac{1}{\ell} \|\varepsilon\|^2 = \frac{1}{\ell} \varepsilon^\top \varepsilon.
$$

Дифференцируя по $\theta$:
$$
\nabla_\theta Q = \frac{2}{\ell} X^\top \varepsilon.
$$

Аналогично, градиент по $\theta_0$:
$$
\frac{\partial Q}{\partial \theta_0} = \frac{2}{\ell} \mathbf{1}^\top \varepsilon.
$$

---

## 3. Эквивалентность с добавлением константного признака

Пусть к каждому объекту $x_i$ добавим фиктивный признак $x_{i,0}=1$. Получим расширенный вектор $\tilde{x}_i = (1, x_{i1},\dots,x_{id})^\top$.  
Расширенный вектор параметров $\tilde{\theta} = (\theta_0, \theta_1,\dots,\theta_d)^\top$.

Тогда скалярное произведение:
$$
\langle \tilde{\theta}, \tilde{x}_i \rangle = \theta_0 \cdot 1 + \sum_{j=1}^{d} \theta_j x_{ij} = \langle \theta, x_i \rangle + \theta_0.
$$

Функционал качества:
$$
Q = \frac{1}{\ell} \sum_{i=1}^{\ell} \bigl( \langle \tilde{\theta}, \tilde{x}_i \rangle - y_i \bigr)^2.
$$

Градиент по $\tilde{\theta}$:
$$
\nabla_{\tilde{\theta}} Q = \frac{2}{\ell} \tilde{X}^\top \varepsilon,
$$
где $\tilde{X}$ – матрица $\ell \times (d+1)$ с первым столбцом из единиц. Первая компонента этого градиента в точности равна $\partial Q / \partial \theta_0$, а остальные – $\partial Q / \partial \theta_j$. Таким образом, добавление единичного столбца позволяет унифицировать запись и автоматически учитывать смещение.

---

## 4. Сравнение полного градиентного спуска и стохастического (SGD)

### Полный градиентный спуск (Batch GD)
- **Шаг**: $\theta \leftarrow \theta - \eta \nabla_\theta Q$, где $\nabla_\theta Q$ вычисляется по **всей** выборке.
- **Точность градиента**: высокая, направление точно соответствует убыванию $Q$.
- **Скорость**: один шаг требует $O(\ell d)$ операций → медленно при больших $\ell$.
- **Сходимость**: монотонное убывание $Q$, гарантированный выход в минимум (для выпуклых функций).

### Стохастический градиентный спуск (SGD)
- **Шаг**: $\theta \leftarrow \theta - \eta \nabla L(\theta, x_i)$, где градиент вычисляется по **одному случайному объекту** (или мини‑батчу).
- **Точность градиента**: низкая, содержит шум (является несмещённой оценкой, но с большой дисперсией).
- **Скорость**: один шаг $O(d)$ → очень быстрый, можно обрабатывать огромные выборки.
- **Приобретается**: возможность онлайн‑обучения, шум помогает «выскакивать» из локальных минимумов, часто быстрее достигает приемлемого решения.
- **Теряется**: монотонность убывания $Q$; требуется уменьшение learning rate ($\eta_t \sim 1/t$) для сходимости; итоговая точность может быть хуже (но для машинного обучения это часто некритично).

На практике используют **мини‑батчевый SGD** (батч размера $B$), который даёт компромисс между скоростью и точностью градиента.
