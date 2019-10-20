import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@SuppressWarnings("AlibabaRemoveCommentedCode")
public class LeetCode {

    public static class ListNode {
        int val;
        @Nullable ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }

    }


    public static int myAtoi(String str) {
        // 去除空格
        str = str.trim();
        // 保证为整数
        boolean matches = str.matches("[-+]?\\d+.*");
        int ans = 0;
        if (matches) {
            // 截取字符串
            String pattern = "[-+]?\\d+";
            Pattern p = Pattern.compile(pattern);
            Matcher matcher = p.matcher(str);
            if (matcher.find()) {
                str = matcher.group();
            }
            // 判断第一位是否为正负号
            char c = str.charAt(0);
            int pos = 0;
            int nag = 0;
            switch (c) {
                case '+':
                    pos = 1;
                    break;
                case '-':
                    nag = 1;
                    break;
                default:
                    break;
            }
            // 第一位为正数
            if (pos == 1) {
                for (int i = 1; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && res > 7)) {
                        return 2147483647;
                    }
                    ans = ans * 10 + res;
                }
            } else if (nag == 1) {
                for (int i = 1; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (-ans < Integer.MIN_VALUE / 10 || (-ans == Integer.MIN_VALUE / 10 && res > 8)) {
                        return -2147483648;
                    }
                    ans = ans * 10 + res;
                }
                ans = ans * (-1);
            } else {
                for (int i = 0; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && res > 7)) {
                        return 2147483647;
                    }
                    ans = ans * 10 + res;
                }
            }
            return ans;
        }
        return 0;
    }

    public static String intToRoman(int num) {
        int[] nums = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] romans = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        int index = 0;
        StringBuilder sb = new StringBuilder();
        while (index < nums.length) {

            while (num >= nums[index]) {
                sb.append(romans[index]);
                num = num - nums[index];
            }
            index++;
        }
        return sb.toString();

    }

    public static int romanToInt(String s) {
        Map<String, Integer> map = new HashMap<>();
        map.put("I", 1);
        map.put("IV", 4);
        map.put("V", 5);
        map.put("IX", 9);
        map.put("X", 10);
        map.put("XL", 40);
        map.put("L", 50);
        map.put("XC", 90);
        map.put("C", 100);
        map.put("CD", 400);
        map.put("D", 500);
        map.put("CM", 900);
        map.put("M", 1000);

        int ans = 0;
        for (int i = 0; i < s.length(); ) {
            if (i + 1 < s.length() && map.containsKey(s.substring(i, i + 2))) {
                ans += map.get(s.substring(i, i + 2));
                i += 2;
            } else {
                ans += map.get(s.substring(i, i + 1));
                i++;
            }
        }
        return ans;
    }

    /**
     * 14. 最长公共前缀
     * 标签：链表
     * 当字符串数组长度为 0 时则公共前缀为空，直接返回
     * 令最长公共前缀 ans 的值为第一个字符串，进行初始化
     * 遍历后面的字符串，依次将其与 ans 进行比较，两两找出公共前缀，最终结果即为最长公共前缀
     * 如果查找过程中出现了 ans 为空的情况，则公共前缀不存在直接返回
     * 时间复杂度：O(s)O(s)，s 为所有字符串的长度之和
     *
     * @param strs
     * @return
     */
    public static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0)
            return "";
        String ans = strs[0];
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            for (; j < ans.length() && j < strs[i].length(); j++) {
                if (ans.charAt(j) != strs[i].charAt(j))
                    break;
            }
            ans = ans.substring(0, j);
            if (ans.equals(""))
                return ans;
        }
        return ans;
    }

    /**
     * 15. 三数之和
     *
     * @param nums
     * @return
     */
    @NotNull
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList();
        int len = nums.length;
        if (nums == null || len < 3) return ans;
        Arrays.sort(nums); // 排序
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0) break;
            // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
            if (i > 0 && nums[i] == nums[i - 1]) continue; // 去重
            int L = i + 1;
            int R = len - 1;
            while (L < R) {
                int sum = nums[i] + nums[L] + nums[R];
                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[L], nums[R]));
                    while (L < R && nums[L] == nums[L + 1]) L++; // 去重
                    while (L < R && nums[R] == nums[R - 1]) R--; // 去重
                    L++;
                    R--;
                } else if (sum < 0) L++;
                else if (sum > 0) R--;
            }
        }
        return ans;
    }

    /**
     * 16. 最接近的三数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(@Nullable int[] nums, int target) {
        if (nums == null || nums.length < 3) {
            return -1;
        }
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length; i++) {
            int start = i + 1, end = nums.length - 1;
            while (start < end) {
                int sum = nums[i] + nums[start] + nums[end];
                if (Math.abs(target - sum) < Math.abs(target - ans)) {
                    ans = sum;
                }
                if (sum > target) {
                    end--;
                } else if (sum < target) {
                    start++;
                } else {
                    return ans;
                }
            }
        }
        return ans;
    }

    /**
     * 17. 电话号码的字母组合
     *
     * @param digits
     * @return
     */
    @NotNull
    public static List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>();
        if (digits.isEmpty()) return ans;
        String[] mapping = new String[]{"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
//            int x = Character.getNumericValue(digits.charAt(i));
            int x = Integer.valueOf(digits.charAt(i) + "");
            while (ans.peek().length() == i) {
                String t = ans.remove();
                for (char s : mapping[x].toCharArray())
                    ans.add(t + s);
            }
        }
        return ans;
    }

    /**
     * 18. 四数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public static List<List<Integer>> fourSum(@Nullable int[] nums, int target) {
        if (nums == null || nums.length < 4) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int start = j + 1, end = nums.length - 1;

                while (start < end) {
                    int temp = nums[i] + nums[j] + nums[start] + nums[end];
                    if (target == temp) {
                        ArrayList<Integer> tList = new ArrayList<>();
                        tList.add(nums[i]);
                        tList.add(nums[j]);
                        tList.add(nums[start]);
                        tList.add(nums[end]);
                        ans.add(tList);
                        start++;
                        end--;
                    } else if (target > temp) {
                        start++;

                    } else {
                        end--;
                    }
                }

            }
        }
        return new ArrayList<>(new HashSet<>(ans));
    }

    /**
     * 19.删除链表的倒数第N个节点
     *
     * @param head
     * @param n
     * @return
     */
    @Nullable
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode p = new ListNode(0);
        p.next = head;
        ListNode first = p;
        ListNode second = p;
        // 提前移动 n+1步
        for (int i = 1; i <= n + 1; i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return p.next;
    }

    /**
     * 20. 有效的括号
     *
     * @param s
     * @return
     */
    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        HashMap<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put('}', '{');
        map.put(']', '[');
        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);
            if (map.containsKey(c)) {
                Character value = stack.empty() ? '#' : stack.pop();
                if (!value.equals(map.get(c))) {
                    return false;
                }
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }

    /**
     * 21. 合并两个有序链表
     *
     * @param l1
     * @param l2
     * @return
     */
    @Nullable
    public static ListNode mergeTwoLists(@Nullable ListNode l1, @Nullable ListNode l2) {

        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }

        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    /**
     * 23. 合并K个排序链表
     *
     * @param lists
     * @return
     */
    @Nullable
    public static ListNode mergeKLists(@Nullable ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }

        return merge(lists, 0, lists.length - 1);


    }

    @Nullable
    static ListNode merge(ListNode[] lists, int left, int right) {
        if (left == right) {
            return lists[left];
        }
        int mid = left + (right - left) / 2;
        ListNode l1 = merge(lists, left, mid);
        ListNode l2 = merge(lists, mid + 1, right);
        return mergeTwoLists(l1, l2);
    }

    /**
     * @param nums
     */
    public void nextPermutation(@NotNull int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i + 1] <= nums[i]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    /**
     * 32 最长有效括号
     *
     * @param s
     * @return
     */
    public static int longestValidParentheses(@Nullable String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int ans = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.empty()) {
                    stack.push(i);
                } else {
                    ans = Math.max(ans, i - stack.peek());
                }
            }
        }
        return ans;
    }

    /**
     * 33. 搜索旋转排序数组
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(@NotNull int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if ((nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo == hi && nums[lo] == target ? lo : -1;
    }

    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置
     *
     * @param nums
     * @param target
     * @param left
     * @return
     */
    private static int extremeInsertionIndex(int[] nums, int target, boolean left) {
        int lo = 0;
        int hi = nums.length;

        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (nums[mid] > target || (left && target == nums[mid])) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        return lo;
    }

    @NotNull
    public static int[] searchRange(@NotNull int[] nums, int target) {
        int[] targetRange = {-1, -1};

        int leftIdx = extremeInsertionIndex(nums, target, true);

        // assert that `leftIdx` is within the array bounds and that `target`
        // is actually in `nums`.
        if (leftIdx == nums.length || nums[leftIdx] != target) {
            return targetRange;
        }

        targetRange[0] = leftIdx;
        targetRange[1] = extremeInsertionIndex(nums, target, false) - 1;

        return targetRange;
    }


    @NotNull
    private List<List<Integer>> res = new ArrayList<>();
    private int[] candidates;
    private int len;

    /**
     * 39. 组合总和
     * 注意：后面选取的数不能比前面选的数还要小
     *
     * @param residue
     * @param start
     * @param pre
     */
    private void findCombinationSum(int residue, int start, @NotNull Stack<Integer> pre) {
        if (residue == 0) {
            // Java 中可变对象是引用传递，因此需要将当前 path 里的值拷贝出来
            res.add(new ArrayList<>(pre));
            return;
        }
        // 优化添加的代码2：在循环的时候做判断，尽量避免系统栈的深度
        // residue - candidates[i] 表示下一轮的剩余，如果下一轮的剩余都小于 0 ，就没有必要进行后面的循环了
        // 这一点基于原始数组是排序数组的前提，因为如果计算后面的剩余，只会越来越小
        for (int i = start; i < len && residue - candidates[i] >= 0; i++) {
            pre.add(candidates[i]);
            // 【关键】因为元素可以重复使用，这里递归传递下去的是 i 而不是 i + 1
            findCombinationSum(residue - candidates[i], i, pre);
            pre.pop();
        }
    }

    @NotNull
    public List<List<Integer>> combinationSum(@NotNull int[] candidates, int target) {
        int len = candidates.length;
        if (len == 0) {
            return res;
        }
        // 优化添加的代码1：先对数组排序，可以提前终止判断
        Arrays.sort(candidates);
        this.len = len;
        this.candidates = candidates;
        findCombinationSum(target, 0, new Stack<>());
        return res;
    }

    /**
     * 42. 接雨水
     *
     * @param height
     * @return
     */
    public static int trap(int[] height) {
        int ans = 0;
        int size = height.length;
        for (int i = 1; i < size - 1; i++) {
            int max_left = 0, max_right = 0;
            for (int j = i; j >= 0; j--) {
                max_left = Math.max(max_left, height[j]);
            }
            for (int j = i; j < size; j++) {
                max_right = Math.max(max_right, height[j]);
            }
            ans += Math.min(max_left, max_right) - height[i];
        }
        return ans;
    }

    /**
     * 46. 全排列
     *
     * @param nums
     * @return
     */
    @NotNull
    public static List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();
        int[] visited = new int[nums.length];
        backtrack(res, nums, new ArrayList<Integer>(), visited);
        return res;

    }

    private static void backtrack(@NotNull List<List<Integer>> res, int[] nums, ArrayList<Integer> tmp, int[] visited) {
        if (tmp.size() == nums.length) {
            res.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] == 1) continue;
            visited[i] = 1;
            tmp.add(nums[i]);
            backtrack(res, nums, tmp, visited);
            visited[i] = 0;
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 48. 旋转图像
     *
     * @param matrix
     */
    public void rotate(@NotNull int[][] matrix) {
        int n = matrix.length;
        // 转置矩阵
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int temp = matrix[j][i];
                matrix[j][i] = matrix[i][j];
                matrix[i][j] = temp;
            }
        }
        // 反转本行
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][n - j - 1];
                matrix[i][n - j - 1] = tmp;
            }
        }
    }

    /**
     * 49. 字母异位词分组
     *
     * @param strs
     * @return
     */
    public static List<List<String>> groupAnagrams(@Nullable String[] strs) {
        if (strs == null || strs.length == 0) return new ArrayList<List<String>>();
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] tmp = str.toCharArray();
            Arrays.sort(tmp);
            String keyStr = String.valueOf(tmp);
            if (!map.containsKey(keyStr)) map.put(keyStr, new ArrayList<String>());
            map.get(keyStr).add(str);
        }
        return new ArrayList<>(map.values());

    }

    /**
     * 53. 最大子序和
     * 暴力法
     *
     * @param nums
     * @return
     */
    public static int maxSubArray(int[] nums) {
        int tmp = nums[0];
        int max = tmp;
        int len = nums.length;
        for (int i = 1; i < len; i++) {
            // 当当前序列加上此时的元素的值大于tmp的值，说明最大序列和可能出现在后续序列中，记录此时的最大值
            if (tmp + nums[i] > nums[i]) {
                max = Math.max(max, tmp + nums[i]);
                tmp = tmp + nums[i];
            } else {   // 当tmp(当前和)小于下一个元素时，当前最长序列到此为止。以该元素为起点继续找最大子序列,
                // 并记录此时的最大
                max = Math.max(Math.max(max, tmp), Math.max(tmp + nums[i], nums[i]));
                tmp = nums[i];
            }
        }
        return max;

    }

    /**
     * 53. 最大子序和
     *
     * @param nums
     * @return
     */
    public int maxSubArray1(@NotNull int[] nums) {
        int ans = nums[0];
        int sum = 0;
        for (int num : nums) {
            if (sum > 0) {
                sum += num;
            } else {
                sum = num;
            }
            ans = Math.max(ans, sum);
        }
        return ans;
    }

//    /**
//     * 55. 跳跃游戏
//     * @param position
//     * @param nums
//     * @return
//     */
//    @Deprecated
//    public boolean canJumpFromPosition(int position, int[] nums) {
//        if (position == nums.length - 1) {
//            return true;
//        }
//
//        int furthestJump = Math.min(position + nums[position], nums.length - 1);
//        for (int nextPosition = position + 1; nextPosition <= furthestJump; nextPosition++) {
//            if (canJumpFromPosition(nextPosition, nums)) {
//                return true;
//            }
//        }
//
//        return false;
//    }
//
//    public boolean canJump(int[] nums) {
//        return canJumpFromPosition(0, nums);
//    }


    /**
     * 55. 跳跃游戏
     *
     * @param nums
     * @return
     */
    public boolean canJump(@NotNull int[] nums) {
        memo = new Index[nums.length];
        // 初始化 memo 的所有元素为 UNKNOWN，
        for (int i = 0; i < memo.length; i++) {
            memo[i] = Index.UNKNOWN;
        }
        // 除了最后一个显然是 GOOD （自己一定可以跳到自己）
        memo[memo.length - 1] = Index.GOOD;
        return canJumpFromPosition(0, nums);
    }

    enum Index {GOOD, BAD, UNKNOWN}

    Index[] memo;

    public boolean canJumpFromPosition(int position, @NotNull int[] nums) {
        if (memo[position] != Index.UNKNOWN) {
            // 如果已知直接返回结果 True / False
            return memo[position] == Index.GOOD;
        }

        int furthestJump = Math.min(position + nums[position], nums.length - 1);
        for (int nextPosition = position + 1; nextPosition <= furthestJump; nextPosition++) {
            if (canJumpFromPosition(nextPosition, nums)) {
                memo[position] = Index.GOOD;
                return true;
            }
        }

        memo[position] = Index.BAD;
        return false;
    }

    /**
     * 56. 合并区间
     *
     * @param intervals
     * @return
     */
    @NotNull
    public int[][] merge(@Nullable int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        if (intervals == null || intervals.length == 0)
            return res.toArray(new int[0][]);

        // Arrays.sort(intervals, (a, b) -> a[0] - b[0]);// a[0] - b[0]大于0就交换顺序
        // 根据二维数组第一个数字大小按每一行整体排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                // TODO Auto-generated method stub
                return o1[0] - o2[0];
            }
        });
        int i = 0;
        while (i < intervals.length) {
            int left = intervals[i][0];
            int right = intervals[i][1];
            // i不能到最后一行,所以要小于(数组的长度 - 1)
            // 判断所在行的right和下一行的left大小,对right重新进行赋最大值,之后再不断进行while循环判断
            while (i < intervals.length - 1 && right >= intervals[i + 1][0]) {
                i++;
                right = Math.max(right, intervals[i][1]);
            }
            res.add(new int[]{left, right});
            i++;
        }
        return res.toArray(new int[0][]);
    }

    /**
     * 62. 不同路径
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) dp[0][i] = 1;
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int minPathSum(@NotNull int[][] grid) {
        int M = grid[0].length;
        int N = grid.length;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }
                if (i == 0 && j > 0) {
                    grid[0][j] += grid[0][j - 1];
                    continue;
                }
                if (j == 0 && i > 0) {
                    grid[i][0] += grid[i - 1][0];
                    continue;
                }
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[N - 1][M - 1];
    }

    /**
     * 70. 爬楼梯
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }

        int[] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;

        for (int i = 2; i < n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n - 1];
    }

    /**
     * 72. 编辑距离
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(@NotNull String word1, @NotNull String word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        int[][] dp = new int[n1 + 1][n2 + 1];
        // 第一行
        for (int j = 1; j <= n2; j++) dp[0][j] = dp[0][j - 1] + 1;
        // 第一列
        for (int i = 1; i <= n1; i++) dp[i][0] = dp[i - 1][0] + 1;

        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
                }
            }
        }
        return dp[n1][n2];
    }

    /**
     * 75. 颜色分类
     */
    public void sortColors(@NotNull int[] nums) {
        int p0 = 0, curr = 0;
        int p2 = nums.length - 1;
        int tmp;
        while (curr <= p2) {
            if (nums[curr] == 0) {
                tmp = nums[p0];
                nums[p0++] = nums[curr];
                nums[curr++] = tmp;
            } else if (nums[curr] == 2) {
                tmp = nums[curr];
                nums[curr] = nums[p2];
                nums[p2--] = tmp;
            } else curr++;
        }
    }

    // 用来标记该节点是否遍历过 因为只允许使用一次
    private boolean[][] marked;
    // 算出左上右下四个点坐标
    //          (x-1,y)
    // (x,y-1)   (x,y)  (x,y+1)
    //          (x+1,y)
    private int[][] directions = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    // 行数
    int m;
    // 列数
    int n;
    String word;
    private char[][] board;

    /**
     * 79. 单词搜索
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        m = board.length;
        if (m == 0) {
            return false;
        }
        n = board[0].length;
        marked = new boolean[m][n];
        this.word = word;
        this.board = board;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean inArea(int x, int y) {
        return x >= 0 && x < m && y >= 0 && y < n;
    }

    private boolean dfs(int i, int j, int start) {
        if (start == word.length() - 1) {
            return board[i][j] == word.charAt(start);
        }
        if (board[i][j] == word.charAt(start)) {
            marked[i][j] = true;
            for (int k = 0; k < 4; k++) {
                int newX = i + directions[k][0];
                int newY = j + directions[k][1];
                if (inArea(newX, newY) && !marked[newX][newY]) {
                    if (dfs(newX, newY, start + 1)) {
                        return true;
                    }
                }

            }
            marked[i][j] = false;
        }
        return false;
    }

    private void find(int[] nums, int begin, @NotNull List<Integer> pre) {
        // 没有显式的递归终止
        res.add(new ArrayList<>(pre));// 注意：Java 的引用传递机制，这里要 new 一下
        for (int i = begin; i < nums.length; i++) {
            pre.add(nums[i]);
            find(nums, i + 1, pre);
            pre.remove(pre.size() - 1);// 组合问题，状态在递归完成后要重置
        }
    }

    /**
     * 78. 子集
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(@NotNull int[] nums) {
        int len = nums.length;
        res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        List<Integer> pre = new ArrayList<>();
        find(nums, 0, pre);
        return res;
    }


    /**
     * 94. 二叉树的中序遍历
     *
     * @param root
     * @return 时间复杂度：O(n)。
     * 空间复杂度：O(n))。
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            res.add(curr.val);
            curr = curr.right;
        }
        return res;
    }

    /**
     * 96. 不同的二叉搜索树
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    /**
     * 98. 验证二叉搜索树
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        return helper(root, null, null);
    }

    public boolean helper(TreeNode node, Integer lower, Integer upper) {
        if (node == null) {
            return true;
        }
        int val = node.val;
        if (lower != null && val <= lower) {
            return false;
        }
        if (upper != null && val >= upper) {
            return false;
        }
        if (!helper(node.left, lower, val)) {
            return false;
        }
        if (!helper(node.right, val, upper)) {
            return false;
        }
        return true;
    }

    /**
     * 101. 对称二叉树
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        return (t1.val == t2.val) && isMirror(t1.left, t2.right) && isMirror(t2.left, t1.right);
    }

    List<List<Integer>> ans = new ArrayList<>();

    /**
     * 102. 二叉树的层次遍历
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return ans;
        }
        helper(root, 0);
        return ans;
    }

    private void helper(TreeNode node, int level) {
        if(ans.size() == level){ans.add(new ArrayList<Integer>());}
        ans.get(level).add(node.val);
        if(node.left != null){helper(node.left,level+1);}
        if(node.right != null){helper(node.right,level+1);}

    }
    /**
     * 102. 二叉树的层次遍历
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder1(TreeNode root) {
        List<List<Integer>> levels = new ArrayList<List<Integer>>();
        if (root == null) return levels;

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        int level = 0;
        while ( !queue.isEmpty() ) {
            // start the current level
            levels.add(new ArrayList<Integer>());

            // number of elements in the current level
            int level_length = queue.size();
            for(int i = 0; i < level_length; ++i) {
                TreeNode node = queue.remove();

                // fulfill the current level
                levels.get(level).add(node.val);

                // add child nodes of the current level
                // in the queue for the next level
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            // go to next level
            level++;
        }
        return levels;
    }

    /**
     * 104. 二叉树的最大深度
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if(root == null){return 0;}
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }

    /**
     * 105. 从前序与中序遍历序列构造二叉树
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int preLen = preorder.length;
        int inLen = inorder.length;
        if (preLen != inLen) {
            throw new RuntimeException("Incorrect input data.");
        }
        return buildTree(preorder, 0, preLen - 1, inorder, 0, inLen - 1);
    }


    /**
     * 使用数组 preorder 在索引区间 [preLeft, preRight] 中的所有元素
     * 和数组 inorder 在索引区间 [inLeft, inRight] 中的所有元素构造二叉树
     *
     * @param preorder 二叉树前序遍历结果
     * @param preLeft  二叉树前序遍历结果的左边界
     * @param preRight 二叉树前序遍历结果的右边界
     * @param inorder  二叉树后序遍历结果
     * @param inLeft   二叉树后序遍历结果的左边界
     * @param inRight  二叉树后序遍历结果的右边界
     * @return 二叉树的根结点
     */
    private TreeNode buildTree(int[] preorder, int preLeft, int preRight,
                               int[] inorder, int inLeft, int inRight) {
        // 因为是递归调用的方法，按照国际惯例，先写递归终止条件
        if (preLeft > preRight || inLeft > inRight) {
            return null;
        }
        // 先序遍历的起点元素很重要
        int pivot = preorder[preLeft];
        TreeNode root = new TreeNode(pivot);
        int pivotIndex = inLeft;
        // 严格上说还要做数组下标是否越界的判断 pivotIndex < inRight
        while (inorder[pivotIndex] != pivot) {
            pivotIndex++;
        }
        root.left = buildTree(preorder, preLeft + 1, pivotIndex - inLeft + preLeft,
                inorder, inLeft, pivotIndex - 1);
        root.right = buildTree(preorder, pivotIndex - inLeft + preLeft + 1, preRight,
                inorder, pivotIndex + 1, inRight);
        return root;
    }
    private TreeNode pre;

    /**
     * 114
     * 二叉树展开为链表
     * @param root
     */
    public void flatten(TreeNode root) {
        if(root == null){return;}
        if(pre != null){
            pre.right = root;
            pre.left = null;
        }
        pre = root;
        flatten(root.left);
        flatten(root.right);
    }

    /**
     * 121. 买卖股票的最佳时机
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice)
                minprice = prices[i];
            else if (prices[i] - minprice > maxprofit)
                maxprofit = prices[i] - minprice;
        }
        return maxprofit;
    }

    public static void main(String[] args) {
//        String[] array = {"abc", "acb", "aq", "qa"};
        int[] nums = {1, 2, 3};
        List<List<Integer>> subsets = new LeetCode().subsets(nums);
        System.out.println(subsets);
    }
}
