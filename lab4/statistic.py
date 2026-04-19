import sys
import matplotlib.pyplot as plt


def read_data(filename):
    """
    Читает файл и возвращает два списка: размеры матриц и времена выполнения.
    Пропускает пустые строки и строки с некорректными данными.
    """
    sizes = []
    times = []
    threads = []
    shared = []
    with open(filename, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # пустая строка
            parts = line.split()
            if len(parts) < 2:
                print(f"Строка {line_num}: пропущена (недостаточно чисел)")
                continue
            try:
                size = float(parts[0])
                time = float(parts[1])
                thread = float(parts[2])
                share = float(parts[3])
                sizes.append(size)
                times.append(time)
                threads.append(thread)
                shared.append(share)
            except ValueError:
                print(f"Строка {line_num}: не удалось преобразовать в числа")
    return sizes, times, threads, shared


def plot_graph(size, times, threads, shared, output_filename="plot"):
    """
    Строит график зависимости времени от размера матрицы и сохраняет в JPG.
    """
    output_filename += str(int(size)) + ("(shared)" if shared else "") + ".jpg"
    plt.figure(figsize=(8, 6))
    plt.plot(threads, times, "o-", markersize=5, linewidth=1, color="b")
    plt.xlabel("Размер блока (квадратного)")
    plt.ylabel("Время выполнения (us)")
    plt.title(f"Зависимость времени умножения {size}*{size} от размеров блоков")
    plt.grid(True)
    plt.savefig(
        (output_filename),
        dpi=300,
        format="jpg",
    )
    plt.close()
    print(f"График сохранён как {output_filename}")


def main():
    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    except Exception as e:
        input_file = "to_plot.txt"
        output_file = "plot"

    try:
        sizes, times, threads, shared = read_data(input_file)
        if not sizes:
            print("Нет данных для построения графика.")
            return
        for i in range(len(sizes) // 3):
            plot_graph(
                sizes[i * 3],
                times[i * 3 : (i + 1) * 3],
                threads[i * 3 : (i + 1) * 3],
                shared[i * 3],
                output_file,
            )
    except FileNotFoundError:
        print(f"Ошибка: файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
