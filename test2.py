import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import PIL
from PIL import Image, ImageDraw

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 确保中文显示正常
plt.rcParams["axes.unicode_minus"] = False

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 手写数字识别应用类
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别系统")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # 初始化模型
        self.model = CNN()
        self.model_path = "mnist_cnn.pth"
        self.is_model_trained = os.path.exists(self.model_path)
        
        # 创建标签页
        self.tab_control = ttk.Notebook(root)
        
        # 训练标签页
        self.train_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.train_tab, text="模型训练")
        
        # 识别标签页
        self.recognize_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.recognize_tab, text="手写识别")
        
        # 结果标签页
        self.results_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.results_tab, text="结果分析")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # 创建各标签页内容
        self.create_train_tab()
        self.create_recognize_tab()
        self.create_results_tab()
        
        # 如果模型已训练，加载模型
        if self.is_model_trained:
            self.load_model()
    
    def create_train_tab(self):
        frame = ttk.Frame(self.train_tab, padding=20)
        frame.pack(fill="both", expand=True)
        
        # 训练参数设置
        ttk.Label(frame, text="训练参数设置", font=("SimHei", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(frame, text="批次大小:").grid(row=1, column=0, sticky="w", pady=5)
        self.batch_size_var = tk.IntVar(value=64)
        ttk.Entry(frame, textvariable=self.batch_size_var, width=10).grid(row=1, column=1, sticky="w", pady=5)
        
        ttk.Label(frame, text="训练轮数:").grid(row=2, column=0, sticky="w", pady=5)
        self.epochs_var = tk.IntVar(value=5)
        ttk.Entry(frame, textvariable=self.epochs_var, width=10).grid(row=2, column=1, sticky="w", pady=5)
        
        ttk.Label(frame, text="学习率:").grid(row=3, column=0, sticky="w", pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(frame, textvariable=self.lr_var, width=10).grid(row=3, column=1, sticky="w", pady=5)
        
        # 训练按钮
        self.train_button = ttk.Button(frame, text="开始训练", command=self.train_model)
        self.train_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        # 模型状态
        self.model_status_var = tk.StringVar()
        if self.is_model_trained:
            self.model_status_var.set("模型状态: 已训练 (点击加载)")
        else:
            self.model_status_var.set("模型状态: 未训练")
        
        ttk.Label(frame, textvariable=self.model_status_var).grid(row=5, column=0, columnspan=2, pady=5)
        
        # 加载模型按钮
        self.load_button = ttk.Button(frame, text="加载模型", command=self.load_model, state=tk.NORMAL if self.is_model_trained else tk.DISABLED)
        self.load_button.grid(row=6, column=0, columnspan=2, pady=5)
        
        # 训练日志
        ttk.Label(frame, text="训练日志:").grid(row=7, column=0, sticky="w", pady=10)
        self.log_text = tk.Text(frame, height=10, width=60)
        self.log_text.grid(row=8, column=0, columnspan=2, sticky="we", pady=5)
        scrollbar = ttk.Scrollbar(frame, command=self.log_text.yview)
        scrollbar.grid(row=8, column=2, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 训练进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=9, column=0, columnspan=2, sticky="we", pady=10)
    
    def create_recognize_tab(self):
        frame = ttk.Frame(self.recognize_tab, padding=20)
        frame.pack(fill="both", expand=True)
        
        # 手写绘图区域
        ttk.Label(frame, text="请在下方区域手写数字 (0-9):", font=("SimHei", 12)).grid(row=0, column=0, columnspan=2, pady=10)
        
        # 创建绘图Canvas
        self.draw_canvas = tk.Canvas(frame, width=280, height=280, bg="white", cursor="crosshair")
        self.draw_canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # 绑定绘图事件
        self.draw_canvas.bind("<B1-Motion>", self.paint)
        self.draw_canvas.bind("<Button-1>", self.start_paint)
        
        # 初始化绘图变量
        self.last_x, self.last_y = None, None
        self.image = PIL.Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # 按钮区域
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(button_frame, text="清除", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="识别", command=self.recognize_digit).pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        ttk.Label(frame, text="识别结果:", font=("SimHei", 14)).grid(row=0, column=1, sticky="w", pady=10)
        
        self.result_frame = ttk.Frame(frame, borderwidth=2, relief="groove", padding=10)
        self.result_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # 显示识别的数字
        self.digit_label = tk.Label(self.result_frame, text="?", font=("SimHei", 72), width=3, height=1)
        self.digit_label.pack(pady=20)
        
        # 显示置信度
        self.confidence_label = tk.Label(self.result_frame, text="置信度: --%", font=("SimHei", 12))
        self.confidence_label.pack(pady=10)
        
        # 概率分布图
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_results_tab(self):
        frame = ttk.Frame(self.results_tab, padding=20)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="模型评估结果", font=("SimHei", 14, "bold")).pack(pady=10)
        
        # 创建结果显示区域
        self.results_frame = ttk.Frame(frame)
        self.results_frame.pack(fill="both", expand=True, pady=20)
        
        # 创建图表
        self.accuracy_fig, self.accuracy_ax = plt.subplots(figsize=(8, 4))
        self.accuracy_canvas = FigureCanvasTkAgg(self.accuracy_fig, master=self.results_frame)
        self.accuracy_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 结果文本
        self.results_text = tk.Text(frame, height=10, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def start_paint(self, event):
        self.last_x, self.last_y = event.x, event.y
    
    def paint(self, event):
        if self.last_x and self.last_y:
            self.draw_canvas.create_line((self.last_x, self.last_y, event.x, event.y), width=10, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line((self.last_x, self.last_y, event.x, event.y), width=10, fill=255)
        
        self.last_x, self.last_y = event.x, event.y
    
    def clear_canvas(self):
        self.draw_canvas.delete("all")
        self.image = PIL.Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.digit_label.config(text="?")
        self.confidence_label.config(text="置信度: --%")
        self.ax.clear()
        self.canvas.draw()
    
    def preprocess_image(self):
        # 调整图像大小为28x28
        img = self.image.resize((28, 28), PIL.Image.Resampling.LANCZOS)
        
        # 转换为numpy数组并归一化
        img_array = np.array(img) / 255.0
        
        # 调整为PyTorch期望的格式 (1, 1, 28, 28)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 标准化处理
        transform = transforms.Normalize((0.1307,), (0.3081,))
        img_tensor = transform(img_tensor)
        
        return img_tensor
    
    def recognize_digit(self):
        if not self.is_model_trained:
            messagebox.showerror("错误", "请先训练或加载模型!")
            return
        
        # 预处理图像
        img_tensor = self.preprocess_image()
        
        # 设置为评估模式
        self.model.eval()
        
        # 进行预测
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.exp(output)
            pred = output.argmax(dim=1, keepdim=True).item()
            confidence = probs.max().item() * 100
        
        # 更新结果显示
        self.digit_label.config(text=str(pred))
        self.confidence_label.config(text=f"置信度: {confidence:.2f}%")
        
        # 更新概率分布图
        self.ax.clear()
        self.ax.bar(range(10), probs.squeeze().numpy())
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("数字")
        self.ax.set_ylabel("概率")
        self.ax.set_title("预测概率分布")
        self.fig.tight_layout()
        self.canvas.draw()
    
    def train_model(self):
        # 获取训练参数
        batch_size = self.batch_size_var.get()
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "开始加载数据...\n")
        self.root.update()
        
        # 加载数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
        
        self.log_text.insert(tk.END, f"数据加载完成，训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}\n")
        self.log_text.insert(tk.END, f"开始训练模型 (批次大小: {batch_size}, 轮数: {epochs}, 学习率: {lr})\n")
        self.root.update()
        
        # 初始化模型和优化器
        self.model = CNN()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练循环
        train_losses = []
        test_accuracies = []
        
        for epoch in range(1, epochs + 1):
            # 训练
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 更新进度条
                progress = 100.0 * batch_idx / len(train_loader) + (epoch - 1) * 100.0 / epochs
                self.progress_var.set(progress)
                self.root.update()
                
                # 每100个批次打印一次日志
                if batch_idx % 100 == 0:
                    log_msg = f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' \
                              f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\n'
                    self.log_text.insert(tk.END, log_msg)
                    self.log_text.see(tk.END)
                    self.root.update()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 测试
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            test_accuracies.append(accuracy)
            
            log_msg = f'\nEpoch {epoch} 完成: 平均损失 = {train_loss:.4f}, 测试准确率 = {accuracy:.2f}%\n\n'
            self.log_text.insert(tk.END, log_msg)
            self.log_text.see(tk.END)
            self.root.update()
        
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        self.is_model_trained = True
        self.model_status_var.set("模型状态: 已训练")
        self.load_button.config(state=tk.NORMAL)
        
        self.log_text.insert(tk.END, f"模型已保存至 {self.model_path}\n")
        self.log_text.see(tk.END)
        
        # 更新结果标签页
        self.update_results_tab(train_losses, test_accuracies, test_loader)
        
        messagebox.showinfo("训练完成", f"模型训练完成!\n最终测试准确率: {accuracy:.2f}%")
    
    def update_results_tab(self, train_losses, test_accuracies, test_loader):
        # 清空结果区域
        self.results_text.delete(1.0, tk.END)
        self.accuracy_ax.clear()
        
        # 绘制训练损失和测试准确率曲线
        epochs = range(1, len(train_losses) + 1)
        
        self.accuracy_ax.plot(epochs, train_losses, 'b-', label='训练损失')
        self.accuracy_ax.set_xlabel('轮数')
        self.accuracy_ax.set_ylabel('损失', color='b')
        self.accuracy_ax.tick_params('y', colors='b')
        
        ax2 = self.accuracy_ax.twinx()
        ax2.plot(epochs, test_accuracies, 'r-', label='测试准确率')
        ax2.set_ylabel('准确率 (%)', color='r')
        ax2.tick_params('y', colors='r')
        
        self.accuracy_ax.set_title('训练损失和测试准确率')
        self.accuracy_fig.tight_layout()
        self.accuracy_canvas.draw()
        
        # 添加结果文本
        self.results_text.insert(tk.END, "训练结果总结:\n\n")
        self.results_text.insert(tk.END, f"训练轮数: {len(epochs)}\n")
        self.results_text.insert(tk.END, f"最终训练损失: {train_losses[-1]:.4f}\n")
        self.results_text.insert(tk.END, f"最终测试准确率: {test_accuracies[-1]:.2f}%\n\n")
        
        # 计算每类的准确率
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).squeeze()
                for i in range(len(target)):
                    label = target[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
        
        self.results_text.insert(tk.END, "每类数字的识别准确率:\n")
        for i in range(10):
            self.results_text.insert(tk.END, f"数字 {i}: {100 * class_correct[i] / class_total[i]:.2f}% ({int(class_correct[i])}/{int(class_total[i])})\n")
    
    def load_model(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")],
            initialdir=os.getcwd()
        )
        
        if file_path:
            try:
                # 加载模型
                self.model = CNN()
                self.model.load_state_dict(torch.load(file_path))
                self.model.eval()
                self.is_model_trained = True
                self.model_status_var.set(f"模型状态: 已加载 ({os.path.basename(file_path)})")
                messagebox.showinfo("成功", f"模型已成功加载: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {str(e)}")
                self.is_model_trained = False
                self.model_status_var.set("模型状态: 未训练")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()    