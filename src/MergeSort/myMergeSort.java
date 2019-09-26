package MergeSort;

public class myMergeSort {
    static int number = 0;

    public static void main(String[] args) {
        int[] a = {26, 5, 918, 108};
        printArray("排序前：", a);
        MergeSort(a);
        printArray("排序后：", a);
    }

    private static void printArray(String pre, int[] a) {
        System.out.print(pre + "\n");
        for (int i = 0; i < a.length; i++)
            System.out.print(a[i] + "\t");
        System.out.println();
    }

    private static void MergeSort(int[] a) {
        // TODO Auto-generated method stub
        System.out.println("开始排序");
        Sort(a, 0, a.length - 1);
    }

    private static void Sort(int[] a, int left, int right) {
        if (left >= right) {
            return;
        }

        int mid = (left + right) / 2;
        //二路归并排序里面有两个Sort，多路归并排序里面写多个Sort就可以了
        Sort(a, left, mid);
        Sort(a, mid + 1, right);
        merge(a, left, mid, right);

    }


    /**
     * 合并数组
     *
     * @param arrays
     * @param left      指向数组第一个元素
     * @param mid      指向数组分隔的元素
     * @param right      指向数组最后的元素
     */
    public static void merge(int[] arrays, int left, int mid, int right) {
        //左边的数组的大小
        int[] leftArray = new int[mid - left];
        //右边的数组大小
        int[] rightArray = new int[right - mid + 1];
        //往这两个数组填充数据
        for (int i = left; i < mid; i++) {
            leftArray[i - left] = arrays[i];
        }
        for (int i = mid; i <= right; i++) {
            rightArray[i - mid] = arrays[i];
        }
        int i = 0, j = 0;
        // arrays数组的第一个元素
        int k = left;
        //比较这两个数组的值，哪个小，就往数组上放
        while (i < leftArray.length && j < rightArray.length) {
            //谁比较小，谁将元素放入大数组中,移动指针，继续比较下一个
            // 等于的情况是保证“稳定”
            if (leftArray[i] <= rightArray[j]) {
                arrays[k] = leftArray[i];
                i++;
                k++;
            } else {
                arrays[k] = rightArray[j];
                j++;
                k++;
            }
        }
        //如果左边的数组还没比较完，右边的数都已经完了，那么将左边的数抄到大数组中(剩下的都是大数字)
        while (i < leftArray.length) {
            arrays[k] = leftArray[i];
            i++;
            k++;
        }
        //如果右边的数组还没比较完，左边的数都已经完了，那么将右边的数抄到大数组中(剩下的都是大数字)
        while (j < rightArray.length) {
            arrays[k] = rightArray[j];
            k++;
            j++;
        }
    }


}