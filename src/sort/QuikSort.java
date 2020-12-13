package sort;

import java.util.Arrays;

public class QuikSort {
    void swap(int a[], int i, int j) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

    void quickSort(int a[], int start, int end) {
        if (start >= end) {
            // 只剩下一个元素直接返回
            return;
        }
        int k = a[start];
        int i = start, j = end;
        while (i != j) {
            while (j > i && a[j] >= k) {
                // 后面的比基准值大，不动,j--,找到比k小的值
                --j;
            }
            // 交换k 和 比k小的值
            swap(a, i, j);
            while (i < j && a[i] <= k) {
                // 找到比k大的值，i++
                i++;
            }
            // 交换
            swap(a, i, j);
        }
        quickSort(a, start, i - 1);
        quickSort(a, i + 1, end);
    }

    public static void main(String[] args) {
        QuikSort quikSort = new QuikSort();
        int[] ints = {3, 2, 1, 3, 45, 0};
        quikSort.quickSort(ints, 0, 5);
        System.out.println(Arrays.toString(ints));
    }
}
