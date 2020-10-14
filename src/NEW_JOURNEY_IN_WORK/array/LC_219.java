package NEW_JOURNEY_IN_WORK.array;

import java.util.HashMap;

public class LC_219 {
    /**
     * 解题思路：hashMap中存入数字及其在数组中最后出现的索引，当前数字在hashMap中时，
     * 判断当前索引值与上一次该数字的索引值的差值与k的大小关系，小于等于k则返回true，
     * 否则更新索引值，循环结束时返回false
     * @param nums
     * @param k
     * @return
     */
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (hashMap.containsKey(nums[i])) {
                if (k >= i - hashMap.get(nums[i])) return true;
            }
            hashMap.put(nums[i], i);
        }
        return false;
    }
    public static void main(String[] args) {
        containsNearbyDuplicate(new int[]{1, 2, 3, 1}, 3);
    }
}
