package NEW_JOURNEY_IN_WORK.array;

/**
 * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
 * 如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
 * 你可以假设数组中无重复元素。
 */
public class LC_35 {

    /**
     * 时间复杂度 O（n） 空间复杂度 O（1）
     * @param nums
     * @param target
     * @return
     */
//    public int searchInsert(int[] nums, int target) {
//        for(int i = 0; i < nums.length;i++){
//            if(nums[i] == target){
//                return i;
//            }
//            if(target < nums[i]){
//                return i;
//            }
//        }
//        return nums.length;
//    }

    /**
     * 优化方式：用二分查找
     * 假设题意是叫你在排序数组中寻找是否存在一个目标值，那么训练有素的读者肯定立马就能想到利用二分法在O(logn) 的时间内找到是否存在目标值。
     * 但这题还多了个额外的条件，即如果不存在数组中的时候需要返回按顺序插入的位置，那我们还能用二分法么？
     * 答案是可以的，我们只需要稍作修改即可。
     * 考虑这个插入的位置 pos，它成立的条件为：
     * nums[pos−1]<target≤nums[pos]
     *
     * 其中 nums 代表排序数组。由于如果存在这个目标值，我们返回的索引也是pos，因此我们可以将两个条件合并得出最后的目标：
     * 「在一个有序数组中找第一个大于等于{target}target 的下标」
     *
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n - 1, ans = n;
        while (left <= right) {
            int mid = ((right - left) >> 1) + left;
            if (target <= nums[mid]) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}
