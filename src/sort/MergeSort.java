package sort;

import java.util.Arrays;

public class MergeSort {
    public static void main(String[] args) {
        MergeSort mergeSort = new MergeSort();
        int[] ints = {3, 2, 3, 4, 5, 5};
        int[] tmp = new int[6];
        mergeSort.mergeSort(ints, 0, 5, tmp);
        System.out.println(Arrays.toString(ints));
        System.out.println(Arrays.toString(tmp));
    }

    /**
     * @param a     待排序数组
     * @param start 开始位置
     * @param end   结束位置
     * @param tmp   临时变量，用于存放合并的数组
     */
    void mergeSort(int[] a, int start, int end, int[] tmp) {
        // start 小于 end的时候才排序，等于表明只有一个值，不用二分
        if (start < end) {
            int mid = start + (end - start) / 2;
            mergeSort(a, start, mid, tmp);
            mergeSort(a, mid + 1, end, tmp);
            // 合并数组
            merge(a, start, mid, end, tmp);
        }

    }

    void merge(int[] a, int start, int mid, int end, int tmp[]) {
        // 合并两个数组
        int pb = 0;
        int p1 = start, p2 = mid + 1;
        while (p1 <= mid && p2 <= end) {
            if (a[p1] < a[p2]) {
                tmp[pb++] = a[p1++];
            } else {
                tmp[pb++] = a[p2++];
            }
        }
        while (p1 <= mid) {
            tmp[pb++] = a[p1++];
        }
        while (p2 <= end) {
            tmp[pb++] = a[p2++];
        }

        // 将排好了顺序的数组写到a中
        for (int i = 0; i < end - start + 1; i++) {
            a[start + i] = tmp[i];
        }
    }
}
