package NEW_JOURNEY_IN_WORK.array;

/**
 * 88. 合并两个有序数组
 * 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
 * 说明:
 *
 * 初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。
 * 你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
 *  
 *
 * 示例:
 *
 * 输入:
 * nums1 = [1,2,3,0,0,0], m = 3
 * nums2 = [2,5,6],       n = 3
 *
 * 输出: [1,2,2,3,5,6]
 *
 * 来源：力扣（LeetCode）
 * 链接：https://leetcode-cn.com/problems/merge-sorted-array
 * 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
 */
public class LC_88 {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        // 如果第二个为空，返回第一个
        if(n == 0 && m > 0){return;}
        // 如果第一个没有元素，第二个有元素，复制第一个
        if(m == 0 && n > 0){
            for(int i = 0; i < n;i++){
                nums1[i] = nums2[i];
            }
            return;
        }
        // 两个都有元素
        int total = m + n -1;
        int p = m -1;
        int q = n -1;
        for(int i = total; i>= 0;--i){
            // 两个节点都不为空
            if(p >=0 && q>=0){
                if(nums1[p]>=nums2[q]){
                    nums1[i] = nums1[p--];
                }else{
                    nums1[i] = nums2[q--];
                }
                // 第二个节点为空
            }else if(p >=0 && q < 0){
                nums1[i]= nums1[p--];
                // 第一个节点为空
            }else if(q >=0 && p < 0){
                nums1[i] = nums2[q--];
            }

        }
    }

    public void merge1(int[] nums1, int m, int[] nums2, int n) {
        int len1 = m - 1;
        int len2 = n - 1;
        int len = m + n - 1;
        while(len1 >= 0 && len2 >= 0) {
            // 注意--符号在后面，表示先进行计算再减1，这种缩写缩短了代码
            nums1[len--] = nums1[len1] > nums2[len2] ? nums1[len1--] : nums2[len2--];
        }
        // 表示将nums2数组从下标0位置开始，拷贝到nums1数组中，从下标0位置开始，长度为len2+1
        System.arraycopy(nums2, 0, nums1, 0, len2 + 1);
    }
}
