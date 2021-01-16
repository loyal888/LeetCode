package sort;

import java.util.Arrays;

public class QuikSort {
    public void quick_sort(int[] nums,int left,int right){
        if(left >= right){return;}
        // 确定分界点
        int i = left -1;
        int j = right + 1;
        int midValue = nums[left+right>>1];
        // 划分区间
        while(i < j){
            // 在左序列，寻找大于等于midValue的值
            do {
                ++i;
            }while(nums[i]<midValue);

            // 在右序列，寻找小于等于midValue的值
            do{
                --j;
            }while(nums[j]>midValue);

            // 交换值
            if(i < j){
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
            }
        }
        quick_sort(nums,left,j);
        quick_sort(nums,j+1,right);
    }

    public static void main(String[] args) {
        QuikSort quikSort = new QuikSort();
        int[] ints = {3, 2, 1, 3, 45, 0};
        quikSort.quick_sort(ints,0,5);

        for(int i: ints){
            System.out.println(i);
        }
    }
}
