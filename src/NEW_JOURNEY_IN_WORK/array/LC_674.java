package NEW_JOURNEY_IN_WORK.array;

/**
 * 674. 最长连续递增序列
 * 给定一个未经排序的整数数组，找到最长且连续的的递增序列，并返回该序列的长度。
 * 输入: [1,3,5,4,7]
 * 输出: 3
 * 解释: 最长连续递增序列是 [1,3,5], 长度为3。
 * 尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为5和7在原数组里被4隔开。
 *
 * 来源：力扣（LeetCode）
 * 链接：https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence
 * 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
 *
 */
public class LC_674 {

    /**
     * 解法一:用一个指针指向数组下标，如果后一个比前一个小，结果变量ans++，
     * 否则就将临时变量tmp置1，表示当前只有一个
     * @param nums
     * @return
     */
    public static int findLengthOfLCIS(int[] nums) {
        // 判空
        if(nums == null || nums.length<=0){
            return 0;
        }
        // 判断只有一个值
        if(nums.length == 1){return 1;}

        // 遍历的index
        int cur = 0;
        // 结果
        int ans = 0;
        // 有几个值，从1开始
        int tmp = 1;
        while (cur < nums.length - 1) {
            if (nums[cur] < nums[cur + 1]) {
                tmp++;
            } else {
                tmp = 1;
            }
            cur++;
            ans = Math.max(tmp, ans);
        }
        return ans;
    }

    /**
     * 解法2：滑动窗口法
     * @param nums
     * @return
     */
    public static int findLengthOfLCIS1(int[] nums) {
        // 判空
        if(nums == null || nums.length<=0){
            return 0;
        }
        int ans = 0, anchor = 0;
        for(int i = 0; i < nums.length; i++){
            if(i > 0 && nums[i-1]>= nums[i]){anchor = i;}
            ans = Math.max(i - anchor + 1,ans);
        }
       return ans;
    }

    public static void main(String[] args) {
        int lengthOfLCIS = findLengthOfLCIS(new int[]{1, 2, 2, 3, 4});
        System.out.println(lengthOfLCIS);
    }
}
